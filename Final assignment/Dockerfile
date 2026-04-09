FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# The base image already has Python, pip, and PyTorch setup.
# We use --break-system-packages to bypass the strict environment lock safely inside the container.
RUN pip install --no-cache-dir --break-system-packages \
    pillow \
    transformers \
    timm 

WORKDIR /app

# Copy the required scripts and model into the container
COPY predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

# Run the prediction script
ENTRYPOINT ["python", "/app/predict.py"]

