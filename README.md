MER2LaTeX — Deep Learning Project
Problem Statement

Math Expression Recognition (MER) là bài toán chuyển đổi ảnh công thức toán học → chuỗi LaTeX.
Đầu vào có thể là ảnh chụp, ảnh scan, công thức in hoặc viết tay.
Mục tiêu là tạo ra một pipeline ổn định để:

Nhận dạng công thức từ ảnh

Sinh LaTeX chính xác

(Tùy chọn) Gửi LaTeX vào SymPy để tính toán tự động

Trong project này, chúng tôi tập trung vào so sánh nhiều kiến trúc Deep Learning để tìm ra mô hình phù hợp nhất cho MER.

Solution Overview

Pipeline MER tổng quát:

Image → Encoder (CNN / ViT) → Decoder (LSTM / Transformer) → LaTeX sequence


Các mô hình được triển khai & so sánh:

CNN + BiLSTM + CTC (baseline OCR)

ResNet / EfficientNet + Attention Decoder

Vision Transformer (ViT) + Transformer Decoder

DONUT-style (OCR-free Vision Transformer)

Ngoài ra, nhóm xây dựng đầy đủ:

Preprocessing & augmentation ảnh

Tokenizer (char/BPE/WordPiece)

Loss, scheduler, optimizer

Evaluation (BLEU, edit-distance, exact match…)

Demo MER → LaTeX bằng Gradio

Training + logging (TensorBoard)

Docker hóa toàn bộ môi trường để đảm bảo reproducibility

Làm việc từ Local Machine
1. Clone project về máy local
git clone https://github.com/<your-username>/mer2latex-deeplearning.git
cd mer2latex-deeplearning

2. Chạy Docker
Build & Start
docker-compose build
docker-compose up -d

Attach VSCode vào container

Mở VSCode

Nhấn Ctrl + Shift + P

Gõ: Dev Containers: Attach to Running Container

Chọn container: mer2latex-container

VSCode sẽ mở môi trường của Docker → chạy & chỉnh code trực tiếp trong container.

3. Chạy Jupyter Lab trong Docker
docker exec -it mer2latex-container bash


Trong container:

jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser


Truy cập từ host:
👉 http://localhost:8888

4. Chạy demo Gradio MER → LaTeX
docker exec -it mer2latex-container bash
python app/gradio_app.py


Sau đó mở:
👉 http://localhost:7860

5. Chỉnh sửa code trên máy local

Sử dụng VSCode Dev Container để chỉnh code trực tiếp trong môi trường Docker.
Không cần copy file thủ công theo bất kỳ hướng nào.

6. Push code lên GitHub
git add .
git commit -m "your message"

# Tạo nhánh mới
git checkout -b "ten_nhanh"
git push origin "ten_nhanh"

# Hoặc push lên nhánh main
git push origin main

Project Structure
mer2latex-deeplearning/
│
├─ src/
│   ├─ models/                # CNN-LSTM, Transformer, ViT, Donut
│   ├─ datasets/
│   ├─ engine/                # train loop, eval loop
│   ├─ utils/                 # tokenizer, augmentation, preprocess
│   └─ train.py               # main training script
│
├─ app/
│   └─ gradio_app.py          # Web demo: MER → LaTeX
│
├─ notebooks/                 # EDA + visualization
├─ data/                      # mounted volume (not pushed to Git)
├─ models/                    # checkpoints
├─ logs/                      # tensorboard logs
│
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .gitignore
├─ .dockerignore
└─ README.md

Tech Stack

PyTorch

CNN / BiLSTM / ResNet / ViT / DONUT

Transformer Decoder & Attention Mechanisms

OpenCV + PIL để xử lý ảnh

Tokenizer (char/BPE/WordPiece)

TensorBoard cho logging

Gradio cho demo

Docker + VSCode Dev Container để đồng nhất môi trường

Deliverables

✔️ Pipeline MER hoàn chỉnh

✔️ So sánh ~4 mô hình Deep Learning khác nhau

✔️ Demo web MER → LaTeX bằng Gradio

✔️ Training reproducible với Docker

✔️ Tài liệu báo cáo + slide

✔️ Tích hợp LaTeX → SymPy để evaluate công thức

Notes

Thay đổi bất kỳ file nào:

Dockerfile

docker-compose.yml

requirements.txt

→ cần build lại Docker:

docker-compose build
docker-compose up -d


Dữ liệu thật (folder data/) không push lên GitHub

Checkpoint mô hình (folder models/) nên lưu bằng Git LFS nếu cần