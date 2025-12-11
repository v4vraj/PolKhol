FROM python:3.10-slim

WORKDIR /app

# minimal system deps for pillow and transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

COPY run_perception.py /app/run_perception.py

# Install CPU-only PyTorch wheel + transformers + pillow
# If this errors on your platform, you can remove version pins or run the script locally
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers pillow

ENTRYPOINT ["python", "/app/run_perception.py"]
