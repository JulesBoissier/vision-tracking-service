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
