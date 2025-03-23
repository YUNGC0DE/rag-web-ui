FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  default-libmysqlclient-dev \
  pkg-config \
  netcat-traditional \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model files during build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v3')"

# Copy entrypoint script first
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set Python path and environment
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.6.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
