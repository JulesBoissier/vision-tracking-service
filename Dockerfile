# Use a lightweight Python base image, pinned to a Debian release. Bullseye's security
# archive no longer serves the packages below, so bullseye images can't be built any more.
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV & Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies. Torch comes from PyTorch's CPU-only index: the service runs on CPU,
# and the default (CUDA) wheels were ~2x slower here for identical predictions, besides
# adding ~2 GB of CUDA libraries. Pinned torch versions resolve to their "+cpu" builds.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# Calibration profiles live in /app/data, the only directory docker-compose mounts a
# volume on. The code under /app comes from the image, so rebuilding updates it.
RUN mkdir -p /app/data
ENV CALIBRATION_DB_URL=sqlite:////app/data/calibration.db

# Copy the entire project into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn main:app --host 0.0.0.0 --port 8000"]
