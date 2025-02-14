#!/bin/bash

# Define model paths
MODEL_PATH_DIFFUSION="/ComfyUI/models/diffusion_models/flux1-dev.safetensors"
MODEL_PATH_VAE="/ComfyUI/models/vae/ae.safetensors"
MODEL_PATH_CLIP="/ComfyUI/models/clip/clip_l.safetensors"
MODEL_PATH_T5="/ComfyUI/models/clip/t5xxl_fp16.safetensors"

# Install the required models (clip, t5, flux1-dev, lora), but only if they don't already exist
if [ ! -f "$MODEL_PATH_DIFFUSION" ]; then
    echo "Downloading flux1-dev model..."
    wget --header="Authorization: Bearer $HUGGINGFACE_API_KEY" -O $MODEL_PATH_DIFFUSION https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
else
    echo "flux1-dev model already exists. Skipping download."
fi

if [ ! -f "$MODEL_PATH_VAE" ]; then
    echo "Downloading VAE model..."
    wget --header="Authorization: Bearer $HUGGINGFACE_API_KEY" -O $MODEL_PATH_VAE https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
else
    echo "VAE model already exists. Skipping download."
fi

if [ ! -f "$MODEL_PATH_CLIP" ]; then
    echo "Downloading CLIP model..."
    wget -O $MODEL_PATH_CLIP "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
else
    echo "CLIP model already exists. Skipping download."
fi

if [ ! -f "$MODEL_PATH_T5" ]; then
    echo "Downloading T5 model..."
    wget -O $MODEL_PATH_T5 "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
else
    echo "T5 model already exists. Skipping download."
fi
