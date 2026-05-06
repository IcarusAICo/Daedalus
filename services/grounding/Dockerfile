FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GROUNDING_DEVICE=cuda
ENV GROUNDING_PORT=8420

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download V2 model weights.
RUN mkdir -p weights && \
    hf download microsoft/OmniParser-v2.0 \
        icon_detect/train_args.yaml icon_detect/model.pt icon_detect/model.yaml \
        icon_caption/config.json icon_caption/generation_config.json icon_caption/model.safetensors \
        --local-dir weights && \
    mv weights/icon_caption weights/icon_caption_florence

COPY *.py .

EXPOSE 8420

CMD ["python3", "server.py"]
