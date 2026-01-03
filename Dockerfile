FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 1. Set working directory
WORKDIR /app

# 2. Cai system dependencies can cho DL
RUN apt-get update && apt-get install -y \
    git curl wget libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy file requirements truoc de tan dung cache
COPY requirements.txt .

# 4. Cai Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toan bo project vao container
COPY . .

# 6. Tao thu muc data / models / logs (neu chua co)
RUN mkdir -p data logs

# 7. Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Cho phep CUDA hoat dong ngay ca khi khong co GPU (se fallback ve CPU)
ENV CUDA_VISIBLE_DEVICES=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 8. Lenh mac dinh
CMD ["python", "-c", "import torch; print(f'MER2LaTeX ready! GPU: {torch.cuda.is_available()}')"]
