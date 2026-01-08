# MER2LATEX Gradio Demo App

## Giới thiệu

Demo app sử dụng Gradio để nhận diện công thức toán học từ ảnh và chuyển đổi sang LaTeX.

## Tính năng

**Chức năng chính:**
- Upload ảnh công thức toán học
- Chọn model đã train (từ thư mục `checkpoints/`)
- Chọn loại preprocessing (IM2LATEX hoặc CROHME)
- Xuất kết quả LaTeX
- Hiển thị công thức đã render bằng SymPy
- Xem ảnh sau preprocessing

## Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy demo

**Cách 1: Dùng script**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

**Cách 2: Chạy trực tiếp**
```bash
python app/gradio_app.py
```

### 3. Truy cập giao diện

Mở trình duyệt và truy cập: **http://localhost:7860**

## Hướng dẫn sử dụng giao diện

1. **Chọn Model**: Chọn checkpoint model từ dropdown (ví dụ: `model_b/20260107_000406/best`)

2. **Chọn Preprocessing Type**:
   - `IM2LATEX`: Cho ảnh công thức được render (PNG từ LaTeX)
   - `CROHME`: Cho ảnh công thức viết tay

3. **Upload ảnh**: Kéo thả hoặc click để upload ảnh công thức toán học

4. **Convert**: Click nút "Convert to LaTeX"

5. **Xem kết quả**:
   - LaTeX text output
   - Công thức được render lại bằng SymPy (để so sánh)
   - Ảnh sau preprocessing

## Cấu trúc dữ liệu

```
checkpoints/
├── model_b/
│   ├── 20260104_131225/
│   │   ├── best.pt          # Checkpoint tốt nhất
│   │   └── latest.pt        # Checkpoint mới nhất
│   └── 20260107_000406/
│       ├── best.pt
│       └── latest.pt
vocab.json                    # Vocabulary file (required)
```

## Ví dụ công thức test

Bạn có thể thử với các công thức như:

- **Phương trình đơn giản**: `x^2 + y^2 = r^2`
- **Phân số**: `\frac{a}{b}`, `\frac{x+1}{y-2}`
- **Căn bậc hai**: `\sqrt{x+y}`, `\sqrt[3]{27}`
- **Tích phân**: `\int_{0}^{\infty} e^{-x^2} dx`
- **Ma trận**: `\begin{bmatrix} a & b \\ c & d \end{bmatrix}`
- **Tổng**: `\sum_{i=1}^{n} i^2`

## Xử lý lỗi

### Lỗi: "No models available"
- Kiểm tra thư mục `checkpoints/` có chứa file `.pt`
- Đảm bảo cấu trúc thư mục đúng format: `checkpoints/model_name/timestamp/best.pt`

### Lỗi: "vocab.json not found"
- Đảm bảo file `vocab.json` tồn tại ở root directory
- Chạy tokenizer để tạo vocab: `python -m src.tokenizer.tokenize`

### Lỗi khi load model
- Kiểm tra model architecture phù hợp với checkpoint
- Hiện tại hỗ trợ Model B (ResNet + Attention)

## Công nghệ sử dụng

- **Gradio**: Web UI framework
- **PyTorch**: Deep learning framework
- **SymPy**: Render LaTeX thành ảnh
- **Matplotlib**: Visualization
- **PIL/OpenCV**: Image processing

## Performance Tips

- **GPU**: Tự động sử dụng GPU nếu có sẵn
- **Model caching**: Model được cache lại sau lần load đầu tiên
- **Batch processing**: Hiện tại xử lý từng ảnh một

## Troubleshooting

Nếu gặp vấn đề, check logs để debug:

```bash
# Chạy với verbose output
python app/gradio_app.py 2>&1 | tee demo.log
```

## Tác giả

MER2LATEX Project - Mathematical Expression Recognition to LaTeX
