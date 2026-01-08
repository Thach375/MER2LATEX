"""
LaTeX Utilities
===============
Functions for cleaning and rendering LaTeX strings.
"""

import re
from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Try to import sympy for LaTeX rendering
try:
    from sympy import latex, sympify
    from sympy.parsing.latex import parse_latex
    import matplotlib
    matplotlib.use('Agg')
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def clean_latex_output(latex_str: str) -> str:
    """
    Clean and fix common issues in LaTeX output from model.
    
    Pipeline:
    1. Space Normalization (regex)
    2. Structural Repair (stack-based brace balancing)
    3. Garbage Removal (heuristic cleanup)
    
    Args:
        latex_str: Raw LaTeX string from model
    
    Returns:
        Cleaned LaTeX string
    """
    if not latex_str or not isinstance(latex_str, str):
        return ''
    
    # ========================================================================
    # STEP 1: SPACE NORMALIZATION (Regex)
    # ========================================================================
    
    # Collapse multiple whitespaces into one
    latex_str = re.sub(r'\s+', r' ', latex_str)
    
    # Remove space between command name and opening brace: \frac { -> \frac{
    latex_str = re.sub(r'\s+(_|\^)', r'\1', latex_str)
    latex_str = re.sub(r'(_|\^)\s+\{', r'\1{', latex_str)
    
    
    # Trim leading/trailing spaces
    latex_str = latex_str.strip()
    
    # ========================================================================
    # STEP 2: STRUCTURAL REPAIR (Stack-based)
    # ========================================================================
    
    # Fix subscript/superscript starting at beginning: _x -> {}_x or ^2 -> {}^2
    if latex_str and latex_str[0] in ('_', '^'):
        latex_str = '{}' + latex_str
    
    # Balance braces and parentheses using stack algorithm
    stack_braces = []
    stack_parens = []
    result_chars = []
    
    for char in latex_str:
        if char == '{':
            stack_braces.append(char)
            result_chars.append(char)
        elif char == '}':
            if stack_braces:
                # Valid closing brace
                stack_braces.pop()
                result_chars.append(char)
            # else: skip this extra closing brace (don't add to result)
        elif char == '(':
            stack_parens.append(char)
            result_chars.append(char)
        elif char == ')':
            if stack_parens:
                # Valid closing parenthesis
                stack_parens.pop()
                result_chars.append(char)
            # else: skip this extra closing parenthesis (don't add to result)
        else:
            result_chars.append(char)
    
    # Add missing closing braces if stack still has unmatched opening braces
    if stack_braces:
        result_chars.extend(['}'] * len(stack_braces))
    
    # Add missing closing parentheses if stack still has unmatched opening parentheses
    if stack_parens:
        result_chars.extend([')'] * len(stack_parens))
    
    latex_str = ''.join(result_chars)
    
    # ========================================================================
    # STEP 3: GARBAGE REMOVAL (Heuristic)
    # ========================================================================
    
    # Remove empty groups: {} {} or { }
    for _ in range(5):
        before = latex_str
        latex_str = re.sub(r'\{\s*\}', r'', latex_str)
        if latex_str == before:
            break
    
    # Remove unmatched \left or \right (simplest approach: remove them)
    # More sophisticated: try to balance, but for robustness we remove
    left_count = latex_str.count(r'\left')
    right_count = latex_str.count(r'\right')
    
    if left_count != right_count:
        # Remove all \left and \right to avoid mismatch
        latex_str = re.sub(r'\\left([(\[\{|])?', r'\1', latex_str)
        latex_str = re.sub(r'\\right([)\]\}|])?', r'\1', latex_str)
    
    # Clean up double spaces again after removals
    latex_str = re.sub(r'\s+', r' ', latex_str)
    latex_str = latex_str.strip()
    
    # Final validation: if result is too short or empty, return empty
    if len(latex_str) < 1:
        return ''
    
    return latex_str


def render_latex_with_sympy(latex_str: str) -> Optional[Image.Image]:
    """
    Render LaTeX string as image using SymPy and Matplotlib.
    Includes automatic cleaning and fixing of LaTeX output.
    
    Args:
        latex_str: LaTeX string to render
    
    Returns:
        PIL Image or None if rendering fails
    """
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        # Clean and fix LaTeX output first
        cleaned_latex = clean_latex_output(latex_str)
        
        if not cleaned_latex or cleaned_latex == '...':
            print(f"[WARNING] LaTeX string is empty or too long after cleaning")
            return None
        
        # Try to parse with SymPy (uses antlr4)
        try:
            expr = parse_latex(cleaned_latex)
            latex_to_render = sympy_latex(expr)
        except Exception as parse_error:
            print(f"[DEBUG] SymPy parse failed, using raw LaTeX: {parse_error}")
            # Fall back to rendering raw cleaned LaTeX
            latex_to_render = cleaned_latex
        
        # Dynamic figure sizing based on LaTeX length
        latex_length = len(latex_to_render)
        if latex_length < 50:
            figsize = (8, 1.5)
            fontsize = 28
        elif latex_length < 100:
            figsize = (12, 2.5)
            fontsize = 24
        elif latex_length < 200:
            figsize = (16, 3.5)
            fontsize = 20
        else:
            figsize = (20, 4.5)
            fontsize = 16
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Render LaTeX
        ax.text(
            0.5, 
            0.5,
            f"${latex_to_render}$",
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True
        )
        
        # Convert to image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        image = Image.open(buf)
        plt.close(fig)
        
        return image
    
    except Exception as e:
        print(f"[WARNING] Failed to render LaTeX with SymPy: {e}")
        print(f"[DEBUG] Original LaTeX: {latex_str[:100]}...")
        print(f"[DEBUG] Cleaned LaTeX: {cleaned_latex[:100] if 'cleaned_latex' in locals() else 'N/A'}...")
        return None
