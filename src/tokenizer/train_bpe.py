import pandas as pd
import sentencepiece as spm
import json
import os

# 1. Đọc dữ liệu công thức LaTeX
formulas = pd.read_csv("data/IM2LATEX/im2latex_formulas.norm.csv", header=None, names=["formula"])
formulas["formula"].to_csv("formulas.txt", index=False, header=False)

# 2. Các lệnh LaTeX cần bảo vệ
user_symbols = [
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon",
    "\\text", "\\mathrm", "\\mathbf", "\\mathbb"
]

# 3. Train SentencePiece BPE
spm.SentencePieceTrainer.Train(
    input="formulas.txt",
    model_prefix="latex_bpe",
    vocab_size=2000,
    character_coverage=1.0,
    model_type="bpe",
    user_defined_symbols=user_symbols,
    treat_whitespace_as_suffix=True
)

# 4. Load lại mô hình để lấy vocab
sp = spm.SentencePieceProcessor(model_file="latex_bpe.model")
tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

# 5. Tạo ánh xạ token -> id 
stoi = {tok: i for i, tok in enumerate(tokens)}

# 6. Xuất thẳng ra vocab.json
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)

# 7. Xóa file phụ để chỉ còn vocab.json
for fname in ["latex_bpe.model", "latex_bpe.vocab", "formulas.txt"]:
    if os.path.exists(fname):
        os.remove(fname)

print("Đã train BPE và lưu duy nhất vocab.json")
