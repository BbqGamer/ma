FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

# Set a working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated Python environment so pip doesn't touch the system Python
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install dependencies.
# The PyTorch base image already provides torch/torchvision, so keep them out of pip.
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|torchaudio)([<=>].*)?$' requirements.txt > requirements-runtime.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt

# Copy the training script and container entrypoint.
COPY train_hcl.py .
COPY entrypoint.sh .
RUN chmod +x /workspace/entrypoint.sh

# Create a directory for the dataset and outputs.
RUN mkdir -p /workspace/data /workspace/runs

# Run both experiments by default when the container starts.
ENTRYPOINT ["/workspace/entrypoint.sh"]
