FROM python:3.11-slim

# 1. Set working directory
WORKDIR /app

# 2. Cài system dependencies cần cho DL + xử lý ảnh
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    # lib cho OpenCV / hiển thị ảnh
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy file requirements trước để tận dụng cache
COPY requirements.txt .

# 4. Cài Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ project vào container
COPY . .

# 6. Tạo thư mục data / models / logs (nếu chưa có)
RUN mkdir -p data models logs

# 7. Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 8. Lệnh mặc định (có thể đổi sau)
# Gợi ý: để simple message, dùng docker-compose để override command
CMD ["python", "-c", "print('MER2LaTeX container is ready!')"]
