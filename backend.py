import os
import time
import json
import uuid
import shutil
import tarfile
import tempfile
import subprocess
import asyncio
import requests
import math

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import boto3

app = FastAPI(
    title="Replicate-API Emulation",
    description="FastAPI layer to mimic Replicate’s inference API using ComfyUI",
    version="0.2"
)

# Global variables to manage the ComfyUI subprocess and idle time.
comfy_process = None
last_request_time = time.time()
IDLE_TIMEOUT = 3600  # one hour

# S3 configuration – supply these as environment variables when running the container.
S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")
S3_FOLDER = "generatedPhotos"
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "your-access-key")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "your-secret-key")

# Path to the workflow JSON template (should be included with the image)
WORKFLOW_FILE = "workflow.json"


# -----------------------------------
# Prediction Request Model
# -----------------------------------
class PredictionRequest(BaseModel):
    # Replicate-like input fields
    model: str = "dev"  # For now only "dev" is supported.
    prompt: str
    num_outputs: int = 1
    aspect_ratio: Optional[str] = None  # e.g. "4:5"; if not provided, default workflow resolution is used.
    output_format: str = "jpg"
    lora_scale: float = 1.08
    guidance_scale: float = 2.75
    num_inference_steps: int = 40
    output_quality: int = 100

    # Optionally allow a seed to be specified; if not, a random one will be generated.
    seed: Optional[int] = None

    # Optional webhook URL to call when generation is complete.
    webhook: Optional[str] = None

    # Optional LoRA object with version and weights URL.
    # e.g. {"version": "user-6370-model-1738686626330", "weights": "https://replicate.delivery/xxx/trained_model.tar"}
    lora: Optional[Dict[str, str]] = None


# -----------------------------------
# ComfyUI Process Management
# -----------------------------------
def start_comfy():
    """Starts ComfyUI as a subprocess if not running."""
    global comfy_process
    if comfy_process is None or comfy_process.poll() is not None:
        # Start ComfyUI in headless mode on port 8188.
        comfy_process = subprocess.Popen(
            ["python3", "/ComfyUI/main.py", "--headless", "--port", "8188"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ComfyUI to start (polling the port)
        for _ in range(30):
            try:
                r = requests.get("http://localhost:8188")
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(1)

def stop_comfy():
    """Terminates the ComfyUI process if it is running."""
    global comfy_process
    if comfy_process and comfy_process.poll() is None:
        comfy_process.terminate()
        comfy_process.wait()
        comfy_process = None

async def comfy_idle_checker():
    """Background task that stops ComfyUI if no requests have been processed for one hour."""
    global last_request_time
    while True:
        await asyncio.sleep(60)
        if time.time() - last_request_time > IDLE_TIMEOUT:
            stop_comfy()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(comfy_idle_checker())


# -----------------------------------
# LoRA Management
# -----------------------------------
def manage_lora(lora_info: Dict[str, str]) -> str:
    """
    Given a LoRA description (with keys "version" and "weights"), ensure that the corresponding
    .safetensors file exists in ComfyUI/models/loras. If not, download, extract, and move it.
    Returns the file name (without path) to use in the workflow.
    """
    lora_version = lora_info.get("version")
    weights_url = lora_info.get("weights")
    target_dir = "/ComfyUI/models/loras"
    os.makedirs(target_dir, exist_ok=True)
    expected_filename = f"{lora_version}.safetensors"
    target_path = os.path.join(target_dir, expected_filename)
    if os.path.exists(target_path):
        return expected_filename

    # Download the archive.
    response = requests.get(weights_url, stream=True)
    if response.status_code != 200:
        raise Exception("Failed to download LoRA archive")
    tmp_archive = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp_archive.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Attempt to extract as tar; if fails, try zip.
    extracted_path = tempfile.mkdtemp()
    try:
        with tarfile.open(tmp_archive.name) as tar:
            tar.extractall(path=extracted_path)
    except Exception:
        import zipfile
        with zipfile.ZipFile(tmp_archive.name, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)

    # Search for a .safetensors file.
    found_file = None
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".safetensors"):
                found_file = os.path.join(root, file)
                break
        if found_file:
            break
    if not found_file:
        raise Exception("LoRA safetensors file not found in archive")
    # Move and rename the file.
    shutil.move(found_file, target_path)
    os.remove(tmp_archive.name)
    shutil.rmtree(extracted_path)
    return expected_filename


# -----------------------------------
# Generation Function
# -----------------------------------
def run_comfy_generation(request: PredictionRequest) -> str:
    """
    Wakes up ComfyUI (if not already running), updates the workflow template with the incoming
    parameters, submits the workflow to ComfyUI via its /prompt API, polls for completion via /history,
    and retrieves the generated image via /view.
    Returns the local path to the generated image file.
    """
    start_comfy()
    global last_request_time
    last_request_time = time.time()

    # Load workflow template.
    with open(WORKFLOW_FILE, "r") as f:
        workflow = json.load(f)

    # --- Update the workflow with incoming parameters ---
    # Update the positive prompt node (node "24").
    if "24" in workflow and "inputs" in workflow["24"]:
        workflow["24"]["inputs"]["text"] = request.prompt

    # Update the LoRA loader node (node "45").
    if "45" in workflow and "inputs" in workflow["45"]:
        # If a LoRA object is provided, download/extract it and update the lora_name.
        if request.lora:
            new_lora_name = manage_lora(request.lora)
            workflow["45"]["inputs"]["lora_name"] = new_lora_name
        # Update both strength_model and strength_clip to the provided lora_scale.
        workflow["45"]["inputs"]["strength_model"] = request.lora_scale
        workflow["45"]["inputs"]["strength_clip"] = request.lora_scale

    # Update the sampler node (node "3") with the number of inference steps.
    if "3" in workflow and "inputs" in workflow["3"]:
        workflow["3"]["inputs"]["steps"] = request.num_inference_steps

    # Update the flux guidance nodes ("28" for positive and "29" for negative guidance).
    if "28" in workflow and "inputs" in workflow["28"]:
        workflow["28"]["inputs"]["guidance"] = request.guidance_scale
    if "29" in workflow and "inputs" in workflow["29"]:
        workflow["29"]["inputs"]["guidance"] = request.guidance_scale

    # Update the EmptyLatentImage node (node "6") if an aspect ratio is provided.
    # Default resolution is 1024 height; compute width from aspect ratio.
    if request.aspect_ratio and "6" in workflow and "inputs" in workflow["6"]:
        try:
            w_ratio, h_ratio = [float(x) for x in request.aspect_ratio.split(":")]
            new_height = 1024
            # Compute width = height * (w_ratio / h_ratio) and round to the nearest multiple of 8.
            raw_width = new_height * (w_ratio / h_ratio)
            new_width = int(round(raw_width / 8) * 8)
            workflow["6"]["inputs"]["width"] = new_width
            workflow["6"]["inputs"]["height"] = new_height
        except Exception as e:
            print("Invalid aspect_ratio format. Using default 1024x1024.")

    # Update the seed node (node "69") if a seed is provided; else generate one.
    if "69" in workflow and "inputs" in workflow["69"]:
        workflow["69"]["inputs"]["seed"] = request.seed if request.seed is not None else int(uuid.uuid4().int % 100000)

    # (Other fields like output_format, num_outputs, or output_quality are not used in the workflow.)

    # Submit the workflow to ComfyUI via its /prompt endpoint.
    prompt_endpoint = "http://localhost:8188/prompt"
    resp = requests.post(prompt_endpoint, json=workflow)
    if resp.status_code != 200:
        raise Exception("Failed to queue comfy prompt")
    result = resp.json()
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise Exception("No prompt ID returned by ComfyUI")

    # Poll the /history endpoint for generation completion.
    history_endpoint = f"http://localhost:8188/history/{prompt_id}"
    for _ in range(60):  # wait up to 60 seconds
        time.sleep(1)
        hist_resp = requests.get(history_endpoint)
        if hist_resp.status_code == 200:
            history = hist_resp.json()
            if prompt_id in history and "outputs" in history[prompt_id]:
                outputs = history[prompt_id]["outputs"]
                # Look for the SaveImage node (e.g., node "62") that contains the generated image.
                for node_id, node_output in outputs.items():
                    if "images" in node_output and node_output["images"]:
                        filename = node_output["images"][0]["filename"]
                        # Retrieve the image via the /view endpoint.
                        view_url = f"http://localhost:8188/view?filename={filename}&subfolder=&type=output"
                        image_resp = requests.get(view_url)
                        if image_resp.status_code == 200:
                            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            with open(temp_image.name, "wb") as f:
                                f.write(image_resp.content)
                            return temp_image.name
    raise Exception("Generation timed out")


# -----------------------------------
# S3 Upload Function
# -----------------------------------
def upload_to_s3(file_path: str) -> str:
    """
    Uploads the given file to S3 (or S3-compatible storage) under the S3_FOLDER.
    Returns the URL of the uploaded file.
    """
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    key = f"{S3_FOLDER}/{os.path.basename(file_path)}"
    s3_client.upload_file(file_path, S3_BUCKET, key)
    s3_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
    return s3_url


# -----------------------------------
# FastAPI Endpoint
# -----------------------------------
@app.post("/predictions")
async def create_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Receives a prediction request, triggers generation via ComfyUI (with workflow updated from the input),
    uploads the result to S3, and (if provided) calls back the client’s webhook with a payload similar to Replicate's.
    """
    try:
        loop = asyncio.get_event_loop()
        image_file = await loop.run_in_executor(None, run_comfy_generation, request)
        s3_url = await loop.run_in_executor(None, upload_to_s3, image_file)
        background_tasks.add_task(os.remove, image_file)

        # If a webhook URL is provided, call it with a payload.
        if request.webhook:
            payload = {
                "id": str(uuid.uuid4()),
                "status": "succeeded",
                "output": [s3_url],
                "metrics": {"predict_time": 0},  # minimal metrics for now
                "logs": "Generation completed"
            }
            try:
                requests.post(request.webhook, json=payload)
            except Exception as e:
                print("Webhook callback failed:", e)

        return {"id": str(uuid.uuid4()), "status": "succeeded", "output": [s3_url]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
