import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 1 Sequence Length Analysis
def tokenize_formula(formula):
    """
    Simple tokenization by splitting on whitespace.
    This is a rough estimate - actual tokenization may differ.
    """
    if pd.isna(formula):
        return []
    return str(formula).split()
    
def plot_label_length_distribution(im2latex_df, crohme_df):
    # Sequence length distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].hist(im2latex_df['seq_len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(im2latex_df['seq_len'].median(), color='red', linestyle='--',
                    label=f"Median: {im2latex_df['seq_len'].median():.0f}")
    axes[0].set_xlabel('Sequence Length (tokens)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IM2LATEX: Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].hist(crohme_df['seq_len'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(crohme_df['seq_len'].median(), color='red', linestyle='--',
                    label=f"Median: {crohme_df['seq_len'].median():.0f}")
    axes[1].set_xlabel('Sequence Length (tokens)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CROHME: Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def percentile_analysis(im2latex_df, crohme_df):  # Percentile analysis
    percentiles = [90, 95, 99, 99.5]

    print("\n=== IM2LATEX Sequence Length Percentiles ===")
    for p in percentiles:
        val = np.percentile(im2latex_df['seq_len'], p)
        coverage = (im2latex_df['seq_len'] <= val).mean() * 100
        print(f"{p}th percentile: {val:.0f} tokens (covers {coverage:.1f}% of data)")

    print("\n=== CROHME Sequence Length Percentiles ===")
    for p in percentiles:
        val = np.percentile(crohme_df['seq_len'], p)
        coverage = (crohme_df['seq_len'] <= val).mean() * 100
        print(f"{p}th percentile: {val:.0f} tokens (covers {coverage:.1f}% of data)")
        
# 2 Token Frequency Analysis
def top_common_tokens(cnt):
    # Top 50 most common tokens
    print("\n=== IM2LATEX: Top 50 Most Common Tokens ===")
    for token, count in cnt.most_common(50):
        print(f"{token:20s} : {count:6,}")
        
def rare_token(im2latex_token_counter, crohme_token_counter):
    # Rare tokens (appearing < 5 times)
    im2latex_rare = {token: count for token, count in im2latex_token_counter.items() if count < 5}
    crohme_rare = {token: count for token, count in crohme_token_counter.items() if count < 5}

    print(f"\nIM2LATEX - Rare tokens (count < 5): {len(im2latex_rare):,}")
    print(f"CROHME   - Rare tokens (count < 5): {len(crohme_rare):,}")

    print("\n=== Sample of Rare Tokens (IM2LATEX) ===")
    for token, count in list(im2latex_rare.items())[:30]:
        print(f"{token:30s} : {count}")
        
import matplotlib.pyplot as plt

def plot_topk_token_frequencies(im2latex_token_counter, crohme_token_counter, top_k=50, y_log_scale=True):
    """
    x: token names
    y: frequency (optionally log scale)
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)

    def _plot(counter, ax, title):
        items = counter.most_common(top_k) if hasattr(counter, "most_common") else sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        tokens = [t for t, c in items]
        counts = [c for t, c in items]

        ax.bar(tokens, counts)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Token", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        if y_log_scale:
            ax.set_yscale("log")  # log scale on frequency
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=70)
        # cho dễ đọc token dài
        ax.set_xticklabels(tokens, ha="right")

    _plot(im2latex_token_counter, axes[0], f"IM2LATEX: Top-{top_k} Token Frequencies (log y)")
    _plot(crohme_token_counter, axes[1], f"CROHME: Top-{top_k} Token Frequencies (log y)")

    plt.tight_layout()
    plt.show()
