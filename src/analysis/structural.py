import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 1 Nesting Depth Analysis

def calculate_nesting_depth(formula):
    """
    Calculate maximum nesting depth of braces in a formula.
    """
    if pd.isna(formula):
        return 0
    
    max_depth = 0
    current_depth = 0
    
    for char in str(formula):
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)
    
    return max_depth

def plot_nesting_depth_distribution(im2latex_df, crohme_df):
    # Nesting depth distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].hist(im2latex_df['nesting_depth'], bins=range(0, im2latex_df['nesting_depth'].max() + 2), 
                edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Nesting Depth', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IM2LATEX: Nesting Depth Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].hist(crohme_df['nesting_depth'], bins=range(0, crohme_df['nesting_depth'].max() + 2),
                edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Nesting Depth', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CROHME: Nesting Depth Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
# 2 Formula Type Classification

def classify_formula(formula):
    """
    Classify formula type based on special tokens.
    """
    if pd.isna(formula):
        return 'other'
    
    formula_str = str(formula).lower()
    types = []
    
    # Calculus
    if any(token in formula_str for token in ['\\int', '\\sum', '\\lim', '\\partial']):
        types.append('calculus')
    
    # Linear Algebra (matrices)
    if any(token in formula_str for token in ['\\begin{matrix', '\\begin{bmatrix', '\\begin{pmatrix']):
        types.append('matrix')
    
    # Fractions
    if '\\frac' in formula_str:
        types.append('fraction')
    
    # Subscript/Superscript heavy
    if formula_str.count('_') + formula_str.count('^') > 3:
        types.append('subscript_heavy')
    
    return ','.join(types) if types else 'simple'