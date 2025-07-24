# Use slim Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies needed by OpenCV and video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Render will set ($PORT)
EXPOSE 10000

# Start the application. Render injects $PORT; default to 10000 for local runs
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} app:app 