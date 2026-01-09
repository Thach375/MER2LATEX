import matplotlib.pyplot as plt
from sympy.parsing.latex import parse_latex
from sympy.printing.latex import latex as sympy_latex

# -----------------------------
# LaTeX gốc (GIỮ NGUYÊN)
# -----------------------------
tex = r"g ( \tilde { f } _ { 2 } Q _ { 3 } - \tilde { f } _ { 3 } Q _ { 2 } ) - g ^ { \prime } Q _ { 1 } = 0 ."


# Parse sang SymPy
expr = parse_latex(tex)

# -----------------------------
# Vẽ bảng
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 2))
ax.axis("off")

headers = ["LaTeX", "Image", "Generated SymPy"]
n_rows = 2
n_cols = 3

cell_w = 1 / n_cols
cell_h = 1 / n_rows

# Khung bảng
for row in range(n_rows):
    for col in range(n_cols):
        ax.add_patch(
            plt.Rectangle(
                (col * cell_w, 1 - (row + 1) * cell_h),
                cell_w,
                cell_h,
                fill=False
            )
        )

# Header
for col, h in enumerate(headers):
    ax.text(
        col * cell_w + cell_w / 2,
        1 - cell_h / 2,
        h,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

# -----------------------------
# DÒNG DỮ LIỆU
# -----------------------------

# ✅ Cột 1: LaTeX GỐC (KHÔNG QUA SYMPY)
ax.text(
    cell_w / 2,
    1 - 1.5 * cell_h,
    f"{tex}",
    ha="center",
    va="center",
    fontsize=14
)

# ✅ Cột 2: Image (render từ SymPy cho đẹp)
ax.text(
    cell_w + cell_w / 2,
    1 - 1.5 * cell_h,
    f"${sympy_latex(expr)}$",
    ha="center",
    va="center",
    fontsize=16
)

# ✅ Cột 3: Generated SymPy
ax.text(
    2 * cell_w + cell_w / 2,
    1 - 1.5 * cell_h,
    str(expr),
    ha="center",
    va="center",
    fontsize=12,
    family="monospace"
)

plt.show()
