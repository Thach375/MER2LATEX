# Document Heading  

📘 MER2LaTeX — Deep Learning Project  

Math Expression Recognition: Image → LaTeX → (Optional) SymPy  

✨ Overview  

Math Expression Recognition (MER) là bài toán chuyển ảnh công thức toán học thành chuỗi LaTeX.  

Đầu vào có thể là:  
• Ảnh chụp bảng/giấy  
• Ảnh scan  
• Công thức in  
• Công thức viết tay  

Mục tiêu của dự án:  
• Trích xuất công thức toán học từ ảnh
• Sinh ra LaTeX chính xác
• (Tuỳ chọn) gửi LaTeX vào SymPy để tính toán
• Xây dựng và so sánh nhiều mô hình Deep Learning

🎯 Project Goals  

• Xây dựng pipeline MER hoàn chỉnh  
• Triển khai và so sánh 4 mô hình Deep Learning  
• Preprocessing + augmentation  
• Đánh giá bằng BLEU, Edit Distance, Exact Match  
• Demo web MER → LaTeX bằng Gradio  
• Môi trường huấn luyện đầy đủ bằng Docker  
• Training reproducible  
• Báo cáo + slide hoàn chỉnh  

🧠 MER Pipeline  
Image → Encoder (CNN / ViT) → Decoder (LSTM / Transformer) → LaTeX Tokens  

🏗️ Implemented Models  

1. CNN + BiLSTM + CTC (baseline OCR)  

• Kiến trúc OCR truyền thống  
• Dùng làm baseline để so sánh  

2. ResNet / EfficientNet + Attention Decoder  

• Encoder cực mạnh  
• Decoder sinh LaTeX bằng attention autoregressive  

3. ViT Encoder + Transformer Decoder  

• Khai thác sức mạnh Vision Transformer  
• Phù hợp với công thức phức tạp, nhiều ký hiệu  

4. DONUT-style (OCR-free)  

• Không cần CTC  
• Encoder Vision Transformer → trực tiếp sinh token  
• Gần giống mô hình của NAVER CLova  

🧰 Additional Components  

• Preprocessing & augmentation ảnh  
• Tokenizer (character / BPE / WordPiece)  
• Optimizer, scheduler, gradient clipping  
• Evaluation: BLEU, Edit-distance, Exact Match  
• Logging bằng TensorBoard  
• Gradio demo MER → LaTeX  
• Docker hoá toàn bộ môi trường  

🐳 Docker & Development Guide  

1. Clone project
```
git clone https://github.com/Thach375/MER2LATEX.git  
cd mer2latex-deeplearning  
```
3. Build & Run Docker
```
docker-compose build  
docker-compose up -d  
```
4. Attach VSCode Dev Container  

Trong VSCode:  
```
• Nhấn Ctrl + Shift + P  
• Chọn: Dev Containers: Attach to Running Container  
• Chọn: mer2latex-container  

→ Chỉnh code trực tiếp trong Docker.  
```
4. Chạy Jupyter Lab
```
docker exec -it mer2latex-container bash  
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser  

Truy cập: http://localhost:8888  
```
5. Chạy Gradio demo
```
docker exec -it mer2latex-container bash  
python app/gradio_app.py  
Truy cập demo: http://localhost:7860  
```
6. Push code lên GitHub
```
git add .  
git commit -m "your message"  
```
-> Tạo nhánh  
```
git checkout -b "ten_nhanh"
git push origin "ten_nhanh"
```
-> Hoặc nhánh main
```
git push origin main
```
📁 Project Structure  
MER2LATEX/  
│
├── src/
│   ├── models/           # CNN-LSTM, Transformer, ViT, Donut  
│   ├── datasets/  
│   ├── engine/           # train loop, eval loop  
│   ├── utils/            # tokenizer, augmentation, preprocess  
│   └── train.py          # main training script  
│  
├── app/  
│   └── gradio_app.py     # MER → LaTeX demo  
│  
├── notebooks/            # EDA + visualization  
├── data/                 # ignored by Git  
├── models/               # checkpoints  
├── logs/                 # TensorBoard logs  
│  
├── requirements.txt  
├── Dockerfile  
├── docker-compose.yml  
├── .gitignore  
├── .dockerignore  
└── README.md  

🧪 Tech Stack

• PyTorch  
• CNN / LSTM / Transformer / ViT  
• OpenCV + PIL  
• BPE / WordPiece tokenizer  
• Gradio  
• TensorBoard  
• Docker + VSCode Dev Container  

📦 Deliverables  
  
• Pipeline MER hoàn chỉnh  
• 4 mô hình Deep Learning để so sánh  
• Demo MER → LaTeX  
• Training reproducible với Docker  
• Notebook EDA  
• Báo cáo + slide  
• Tích hợp LaTeX → SymPy  

🔧 Notes  

• Khi thay đổi Dockerfile, docker-compose.yml, requirements.txt → cần build lại Docker  
```
docker-compose build  
docker-compose up -d  
```
• Không push dữ liệu thật → folder data/ đã được ignore  
• Checkpoints nặng → nên dùng Git LFS  
