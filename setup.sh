#!/bin/bash

# Install the required models (clip, t5, flux1-dev, lora)
mkdir -p /ComfyUI/models/vae && \
    mkdir -p /ComfyUI/models/diffusion_models && \
    mkdir -p /ComfyUI/models/loras

# Download the models (only at runtime), using the API key from the environment variable
wget --header="Authorization: Bearer $HUGGINGFACE_API_KEY" -O /ComfyUI/models/diffusion_models/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors

wget --header="Authorization: Bearer $HUGGINGFACE_API_KEY" -O /ComfyUI/models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors

wget -O /ComfyUI/models/clip/clip_l.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"

wget -O /ComfyUI/models/clip/t5xxl_fp16.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
