# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables to prevent pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for the dashboard and optional CV extras
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for caching layers
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Create directory structure for SQLite database and evidence images
RUN mkdir -p data/evidence data/parking_coords

# Expose the dashboard port
EXPOSE 8501

# Command to run the application
CMD ["python", "app.py"]
