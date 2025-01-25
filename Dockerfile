# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV & Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Git-based dependencies first
# RUN pip install git+https://github.com/elliottzheng/face-detection@786fbab7095623c348e251f1f0a8b323721c6a84
RUN pip install git+https://github.com/edavalosanaya/L2CS-Net.git@4a0f978d5b4c426a7d37022d8c927d6ea031dcb6

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Write the databse file
RUN touch /app/calibration.db

# Copy the entire project into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn main:app --host 0.0.0.0 --port 8000"]
