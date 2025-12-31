import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd


# -----------------------------
# 1) Normalize LaTeX
# -----------------------------
def normalize_latex(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()

    # strip $...$ if present
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# 2) LaTeX-aware rule-based tokenizer
# -----------------------------
TOKEN_RE = re.compile(
    r"""
    (\\[a-zA-Z]+)            |  # \command
    (\\[^a-zA-Z])            |  # escaped char like \{ \} \_ \%
    ([{}^_])                 |  # structural tokens
    ([0-9])                  |  # single digit only
    ([a-zA-Z])               |  # single latin letter
    ([+\-*/=<>|:;,.!?()])    |  # operators/punct (includes '.' for decimals)
    (\[|\])                  |  # brackets
    (&)                         # alignment
    """,
    re.VERBOSE,
)

def tokenize_latex(s: str) -> list:
    s = normalize_latex(s)
    if not s:
        return []
    return [m.group(0) for m in TOKEN_RE.finditer(s)]


# -----------------------------
# 3) Build vocab
# -----------------------------
def build_vocab(
    token_lists,
    max_size=1500,
    min_freq=1,
    special_tokens=("<pad>", "<bos>", "<eos>", "<unk>"),
):
    cnt = Counter(t for toks in token_lists for t in toks)

    items = [(tok, f) for tok, f in cnt.items() if f >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    vocab_tokens = [tok for tok, _ in items]
    if max_size is not None:
        keep = max(max_size - len(special_tokens), 0)
        vocab_tokens = vocab_tokens[:keep]

    stoi = {tok: i for i, tok in enumerate(list(special_tokens) + vocab_tokens)}
    return stoi, cnt


# -----------------------------
# 4) Tokenizer class for training
# -----------------------------
class LaTeXTokenizer:
    """
    LaTeX tokenizer for encoding/decoding sequences.
    """
    def __init__(self, vocab_path=None):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.stoi = json.load(f)
        else:
            self.stoi = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3
            }
        
        self.itos = {v: k for k, v in self.stoi.items()}
        
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]
    
    @property
    def vocab_size(self):
        return len(self.stoi)
    
    def encode(self, text, add_bos=True, add_eos=True):
        """Encode text to token ids."""
        tokens = tokenize_latex(text)
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids, skip_special=True):
        """Decode token ids to text."""
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            if i == self.eos_id:
                break
            tokens.append(self.itos.get(i, self.unk_token))
        
        return ' '.join(tokens)
    
    def batch_encode(self, texts, max_length=None, add_bos=True, add_eos=True):
        """Encode batch of texts with padding."""
        encoded = [self.encode(t, add_bos, add_eos) for t in texts]
        
        if max_length is None:
            max_length = max(len(e) for e in encoded)
        
        # Pad sequences
        padded = []
        for seq in encoded:
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [self.pad_id] * (max_length - len(seq))
            padded.append(seq)
        
        return padded


def main():
    from src.utils.constants import IM2LATEX_TRAIN_CSV, CROHME_CSV_PATH
    
    # Read formulas
    im2 = pd.read_csv(IM2LATEX_TRAIN_CSV)["formula"]
    cro = pd.read_csv(CROHME_CSV_PATH)["formula"]

    formulas = pd.concat([im2, cro], ignore_index=True).map(normalize_latex)
    formulas = formulas[formulas.str.len() > 0]

    token_lists = [tokenize_latex(s) for s in formulas.tolist()]

    stoi, cnt = build_vocab(token_lists, max_size=1500, min_freq=1)

    Path("vocab.json").write_text(
        json.dumps(stoi, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # sanity checks
    top20 = cnt.most_common(20)
    suspicious = [t for t in stoi.keys() if ("{" in t or "}" in t) and t not in ["{", "}"]]

    print("[OK] Saved vocab.json")
    print("Vocab size:", len(stoi))
    print("Has \\frac:", "\\frac" in stoi, "| Has \\sqrt:", "\\sqrt" in stoi)
    print("Has { } ^ _:", all(t in stoi for t in ["{", "}", "^", "_"]))
    print("Top 20 tokens:", top20)
    print("Suspicious fused tokens:", suspicious[:20], "| count:", len(suspicious))


if __name__ == "__main__":
    main()
