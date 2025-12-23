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

def tokenize_latex(s: str) -> list[str]:
    s = normalize_latex(s)
    if not s:
        return []
    return [m.group(0) for m in TOKEN_RE.finditer(s)]


# -----------------------------
# 3) Build vocab
# -----------------------------
def build_vocab(
    token_lists,
    max_size=1500,           # gợi ý cho rule-based
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


def main():
    # Read ONLY first column (formula). CSV has: [formula, image_name]
    im2 = pd.read_csv(
        "data/IM2LATEX/label/im2latex_train.csv",
        header=None,
        usecols=[0],
        names=["formula"],
    )["formula"]

    cro = pd.read_csv(
        "data/CROHME/ground_truth/dataset.csv",
        header=None,
        usecols=[0],
        names=["formula"],
    )["formula"]

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

    print("✅ Saved vocab.json")
    print("Vocab size:", len(stoi))
    print("Has \\frac:", "\\frac" in stoi, "| Has \\sqrt:", "\\sqrt" in stoi)
    print("Has { } ^ _:", all(t in stoi for t in ["{", "}", "^", "_"]))
    print("Top 20 tokens:", top20)
    print("Suspicious fused tokens:", suspicious[:20], "| count:", len(suspicious))


if __name__ == "__main__":
    main()
