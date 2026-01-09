"""
LaTeX Utilities
===============
Functions for cleaning and rendering LaTeX strings.
"""

import re
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO

# Configure matplotlib for non-interactive backend FIRST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure mathtext for LaTeX-like rendering
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern (LaTeX default)
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rcParams['font.family'] = 'serif'


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


def _prepare_latex_for_mathtext(latex_str: str) -> str:
    """
    Convert LaTeX string to be compatible with matplotlib's mathtext.
    Mathtext has limited LaTeX support, so we need to convert/remove unsupported commands.
    
    Args:
        latex_str: LaTeX string
    
    Returns:
        Mathtext-compatible LaTeX string
    """
    if not latex_str:
        return latex_str
    
    result = latex_str
    
    # Remove extra spaces around braces and operators for cleaner rendering
    result = re.sub(r'\s*\{\s*', '{', result)
    result = re.sub(r'\s*\}\s*', '}', result)
    result = re.sub(r'\s+', ' ', result)
    
    # Replace unsupported commands with supported alternatives
    replacements = {
        r'\tilde': r'\widetilde',  # tilde -> widetilde
        r'\prime': "'",            # prime -> apostrophe
        r'\displaystyle': '',      # remove displaystyle
        r'\textstyle': '',         # remove textstyle
        r'\scriptstyle': '',       # remove scriptstyle
        r'\scriptscriptstyle': '', # remove scriptscriptstyle
        r'\left': '',              # remove \left
        r'\right': '',             # remove \right
        r'\bigl': '',              # remove sizing
        r'\bigr': '',
        r'\Bigl': '',
        r'\Bigr': '',
        r'\biggl': '',
        r'\biggr': '',
        r'\Biggl': '',
        r'\Biggr': '',
        r'\middle': '',
        r'\Big': '',
        r'\big': '',
        r'\text': r'\mathrm',      # text -> mathrm
        r'\textbf': r'\mathbf',    # textbf -> mathbf  
        r'\textit': r'\mathit',    # textit -> mathit
        r'\bm': r'\mathbf',        # bm -> mathbf
        r'\boldsymbol': r'\mathbf',
        r'\hspace': '',            # remove hspace
        r'\vspace': '',            # remove vspace
        r'\quad': ' ',             # quad -> space
        r'\qquad': '  ',           # qquad -> double space
        r'\;': ' ',                # thick space -> space
        r'\:': ' ',                # medium space -> space
        r'\,': '',                 # thin space -> nothing
        r'\!': '',                 # negative thin space -> nothing
    }
    
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()
    
    return result


def _render_single_latex(latex_str: str) -> Optional[Image.Image]:
    """
    Internal function to render a single LaTeX string to image.
    
    Args:
        latex_str: LaTeX string to render (already cleaned or raw)
    
    Returns:
        PIL Image or None if rendering fails
    """
    if not latex_str or not latex_str.strip():
        return None
    
    try:
        # Prepare LaTeX for mathtext (convert unsupported commands)
        prepared_latex = _prepare_latex_for_mathtext(latex_str)
        
        print(f"[DEBUG] Prepared LaTeX for rendering: {prepared_latex[:60]}...")
        
        # Dynamic figure sizing based on LaTeX length
        latex_length = len(prepared_latex)
        if latex_length < 30:
            figsize = (6, 1.2)
            fontsize = 22
        elif latex_length < 60:
            figsize = (10, 1.8)
            fontsize = 20
        elif latex_length < 120:
            figsize = (14, 2.5)
            fontsize = 18
        else:
            figsize = (18, 3.5)
            fontsize = 16
        
        # Create figure with white background
        fig = plt.figure(figsize=figsize, facecolor='white', edgecolor='none')
        
        # Remove all axes/borders
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Render LaTeX using matplotlib's mathtext
        latex_display = r"$" + prepared_latex + r"$"
        
        text_obj = ax.text(
            0.5, 0.5,
            latex_display,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha='center',
            va='center',
            color='black'
        )
        
        # Render to buffer
        buf = BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=200,
            bbox_inches='tight',
            pad_inches=0.15,
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        buf.seek(0)
        image = Image.open(buf).copy()  # Copy to detach from buffer
        buf.close()
        
        return image
        
    except Exception as e:
        print(f"[ERROR] Matplotlib render failed: {e}")
        plt.close('all')
        return None


def render_latex_with_sympy(latex_str: str) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Render LaTeX string as images using Matplotlib's mathtext renderer.
    Returns two images: one for raw LaTeX, one for cleaned LaTeX.
    
    Args:
        latex_str: Raw LaTeX string from model
    
    Returns:
        Tuple of (raw_image, cleaned_image):
        - raw_image: Rendered image of original latex_str
        - cleaned_image: Rendered image of clean_latex_output(latex_str)
    """
    if not latex_str:
        print("[WARNING] Empty LaTeX string provided")
        return None, None
    
    # Clean the LaTeX
    cleaned_latex = clean_latex_output(latex_str)
    
    print(f"[DEBUG] Raw LaTeX: {latex_str[:80]}...")
    print(f"[DEBUG] Cleaned LaTeX: {cleaned_latex[:80]}...")
    
    # Render raw LaTeX
    raw_image = None
    try:
        raw_image = _render_single_latex(latex_str)
        if raw_image:
            print("[INFO] Successfully rendered raw LaTeX")
        else:
            print("[WARNING] Failed to render raw LaTeX")
    except Exception as e:
        print(f"[ERROR] Raw LaTeX render error: {e}")
    
    # Render cleaned LaTeX
    cleaned_image = None
    try:
        cleaned_image = _render_single_latex(cleaned_latex)
        if cleaned_image:
            print("[INFO] Successfully rendered cleaned LaTeX")
        else:
            print("[WARNING] Failed to render cleaned LaTeX")
    except Exception as e:
        print(f"[ERROR] Cleaned LaTeX render error: {e}")
    
    return raw_image, cleaned_image
