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
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Replicate-API Emulation", 
    description="FastAPI layer to mimic Replicate's inference API using ComfyUI",
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

logger.info(f"Initialized with S3 bucket: {S3_BUCKET}, endpoint: {S3_ENDPOINT}")

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
    logger.info("Attempting to start ComfyUI process...")
    
    if comfy_process is None or comfy_process.poll() is not None:
        # Start ComfyUI in headless mode on port 8188.
        logger.debug("Starting ComfyUI subprocess...")
        comfy_process = subprocess.Popen(
            ["python3", "/ComfyUI/main.py", "--headless", "--port", "8188"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ComfyUI to start (polling the port)
        for attempt in range(30):
            try:
                r = requests.get("http://localhost:8188")
                if r.status_code == 200:
                    logger.info("ComfyUI started successfully")
                    break
            except Exception as e:
                logger.debug(f"Attempt {attempt+1}: Waiting for ComfyUI to start... ({str(e)})")
                time.sleep(1)
        else:
            logger.error("Failed to start ComfyUI after 30 attempts")
            raise Exception("ComfyUI failed to start")

def stop_comfy():
    """Terminates the ComfyUI process if it is running."""
    global comfy_process
    if comfy_process and comfy_process.poll() is None:
        logger.info("Stopping ComfyUI process...")
        comfy_process.terminate()
        comfy_process.wait()
        comfy_process = None
        logger.info("ComfyUI process stopped")

async def comfy_idle_checker():
    """Background task that stops ComfyUI if no requests have been processed for one hour."""
    global last_request_time
    logger.info("Starting idle checker background task")
    while True:
        await asyncio.sleep(60)
        idle_time = time.time() - last_request_time
        logger.debug(f"Current idle time: {idle_time:.2f} seconds")
        if idle_time > IDLE_TIMEOUT:
            logger.info(f"ComfyUI has been idle for {idle_time:.2f} seconds, shutting down")
            stop_comfy()

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up, initializing background tasks...")
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
    logger.info(f"Managing LoRA for version: {lora_info.get('version')}")
    
    lora_version = lora_info.get("version")
    weights_url = lora_info.get("weights")
    target_dir = "/ComfyUI/models/loras"
    os.makedirs(target_dir, exist_ok=True)
    expected_filename = f"{lora_version}.safetensors"
    target_path = os.path.join(target_dir, expected_filename)
    
    if os.path.exists(target_path):
        logger.info(f"LoRA file already exists at {target_path}")
        return expected_filename

    logger.info(f"Downloading LoRA from {weights_url}")
    # Download the archive.
    response = requests.get(weights_url, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to download LoRA archive: {response.status_code}")
        raise Exception("Failed to download LoRA archive")
        
    tmp_archive = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp_archive.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"LoRA archive downloaded to {tmp_archive.name}")

    # Attempt to extract as tar; if fails, try zip.
    extracted_path = tempfile.mkdtemp()
    logger.info(f"Extracting archive to {extracted_path}")
    try:
        with tarfile.open(tmp_archive.name) as tar:
            tar.extractall(path=extracted_path)
            logger.debug("Successfully extracted tar archive")
    except Exception as e:
        logger.debug(f"Tar extraction failed ({str(e)}), attempting zip extraction")
        import zipfile
        with zipfile.ZipFile(tmp_archive.name, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)
            logger.debug("Successfully extracted zip archive")

    # Search for a .safetensors file.
    found_file = None
    logger.info("Searching for .safetensors file in extracted archive")
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".safetensors"):
                found_file = os.path.join(root, file)
                logger.info(f"Found safetensors file: {found_file}")
                break
        if found_file:
            break
            
    if not found_file:
        logger.error("No .safetensors file found in archive")
        raise Exception("LoRA safetensors file not found in archive")
        
    # Move and rename the file.
    logger.info(f"Moving {found_file} to {target_path}")
    shutil.move(found_file, target_path)
    os.remove(tmp_archive.name)
    shutil.rmtree(extracted_path)
    logger.info("LoRA file successfully installed")
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
    logger.info(f"Starting generation for prompt: {request.prompt[:50]}...")
    
    start_comfy()
    global last_request_time
    last_request_time = time.time()

    # Load workflow template.
    logger.debug(f"Loading workflow template from {WORKFLOW_FILE}")
    with open(WORKFLOW_FILE, "r") as f:
        workflow = json.load(f)

    # --- Update the workflow with incoming parameters ---
    logger.info("Updating workflow parameters...")
    
    # Update the positive prompt node (node "24").
    if "24" in workflow and "inputs" in workflow["24"]:
        workflow["24"]["inputs"]["text"] = request.prompt
        logger.debug(f"Set positive prompt: {request.prompt[:50]}...")

    # Update the LoRA loader node (node "45").
    if "45" in workflow and "inputs" in workflow["45"]:
        if request.lora:
            logger.info("Processing custom LoRA...")
            new_lora_name = manage_lora(request.lora)
            workflow["45"]["inputs"]["lora_name"] = new_lora_name
        workflow["45"]["inputs"]["strength_model"] = request.lora_scale
        workflow["45"]["inputs"]["strength_clip"] = request.lora_scale
        logger.debug(f"Set LoRA scale: {request.lora_scale}")

    # Update the sampler node (node "3") with the number of inference steps.
    if "3" in workflow and "inputs" in workflow["3"]:
        workflow["3"]["inputs"]["steps"] = request.num_inference_steps
        logger.debug(f"Set inference steps: {request.num_inference_steps}")

    # Update the flux guidance nodes
    if "28" in workflow and "inputs" in workflow["28"]:
        workflow["28"]["inputs"]["guidance"] = request.guidance_scale
    if "29" in workflow and "inputs" in workflow["29"]:
        workflow["29"]["inputs"]["guidance"] = request.guidance_scale
    logger.debug(f"Set guidance scale: {request.guidance_scale}")

    # Update the EmptyLatentImage node (node "6") if an aspect ratio is provided.
    if request.aspect_ratio and "6" in workflow and "inputs" in workflow["6"]:
        try:
            w_ratio, h_ratio = [float(x) for x in request.aspect_ratio.split(":")]
            new_height = 1024
            raw_width = new_height * (w_ratio / h_ratio)
            new_width = int(round(raw_width / 8) * 8)
            workflow["6"]["inputs"]["width"] = new_width
            workflow["6"]["inputs"]["height"] = new_height
            logger.info(f"Set dimensions to {new_width}x{new_height} for aspect ratio {request.aspect_ratio}")
        except Exception as e:
            logger.warning(f"Invalid aspect_ratio format: {str(e)}. Using default 1024x1024.")

    # Update the seed
    if "69" in workflow and "inputs" in workflow["69"]:
        seed_value = request.seed if request.seed is not None else int(uuid.uuid4().int % 100000)
        workflow["69"]["inputs"]["seed"] = seed_value
        logger.debug(f"Set seed: {seed_value}")

    # Submit the workflow to ComfyUI
    logger.info("Submitting workflow to ComfyUI...")
    prompt_endpoint = "http://localhost:8188/prompt"
    resp = requests.post(prompt_endpoint, json={"prompt": workflow})
    if resp.status_code != 200:
        logger.error(f"Failed to queue comfy prompt: {resp.status_code} with text: {resp.text}")
        raise Exception("Failed to queue comfy prompt")
        
    result = resp.json()
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        logger.error("No prompt ID returned by ComfyUI")
        raise Exception("No prompt ID returned by ComfyUI")
    
    logger.info(f"Generation started with prompt ID: {prompt_id}")

    # Poll for completion
    logger.info("Polling for generation completion...")
    history_endpoint = f"http://localhost:8188/history/{prompt_id}"
    for attempt in range(60):
        time.sleep(1)
        hist_resp = requests.get(history_endpoint)
        if hist_resp.status_code == 200:
            history = hist_resp.json()
            if prompt_id in history and "outputs" in history[prompt_id]:
                outputs = history[prompt_id]["outputs"]
                for node_id, node_output in outputs.items():
                    if "images" in node_output and node_output["images"]:
                        filename = node_output["images"][0]["filename"]
                        logger.info(f"Image generated: {filename}")
                        
                        # Retrieve the image
                        view_url = f"http://localhost:8188/view?filename={filename}&subfolder=&type=output"
                        image_resp = requests.get(view_url)
                        if image_resp.status_code == 200:
                            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            with open(temp_image.name, "wb") as f:
                                f.write(image_resp.content)
                            logger.info(f"Image saved to temporary file: {temp_image.name}")
                            return temp_image.name
                            
        logger.debug(f"Generation still in progress... (attempt {attempt+1}/60)")
    
    logger.error("Generation timed out after 60 seconds")
    raise Exception("Generation timed out")


# -----------------------------------
# S3 Upload Function
# -----------------------------------
def upload_to_s3(file_path: str) -> str:
    """
    Uploads the given file to S3 (or S3-compatible storage) under the S3_FOLDER.
    Returns the URL of the uploaded file.
    """
    logger.info(f"Uploading {file_path} to S3...")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    
    key = f"{S3_FOLDER}/{os.path.basename(file_path)}"
    try:
        s3_client.upload_file(file_path, S3_BUCKET, key)
        s3_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
        logger.info(f"Successfully uploaded to {s3_url}")
        return s3_url
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        raise


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
