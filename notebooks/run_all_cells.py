"""
End-to-end verification script: extracts all code cells from the notebook and
executes them sequentially in a single interpreter context, using a matplotlib
non-interactive backend so no GUI windows appear.
Reports any cell that raises an exception.
"""
import json, sys, os, traceback, io

# Use non-interactive matplotlib backend before any imports
os.environ["MPL_BACKEND"] = "Agg"

NB_PATH = r"c:\Users\DHANRAJ\fraudlens\notebooks\01_eda.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [
    (i, c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
    for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "code"
]

print(f"Found {len(code_cells)} code cells to execute.\n")

# Patch matplotlib backend in the first import cell
global_ns = {
    "__name__": "__main__",
    "__builtins__": __builtins__,
}

# Inject display() shim (available in Jupyter but not plain Python)
def display(*args, **kwargs):
    for a in args:
        print(a)

global_ns["display"] = display

errors = []
for cell_idx, (nb_cell_idx, src) in enumerate(code_cells):
    # Force Agg backend before matplotlib import
    patched_src = src.replace(
        "import matplotlib\n",
        "import matplotlib\nmatplotlib.use('Agg')\n"
    )
    # Also patch plt.show() to be a no-op
    patched_src = patched_src.replace("plt.show()", "plt.close('all')  # plt.show() -> suppressed in CI")

    sys.stdout.write(f"  Executing cell {nb_cell_idx:02d} (code cell #{cell_idx+1:02d}/{len(code_cells)}) ... ")
    sys.stdout.flush()
    try:
        exec(compile(patched_src, f"<cell_{nb_cell_idx}>", "exec"), global_ns)
        print("OK")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR")
        print(f"    {type(e).__name__}: {e}")
        print("    Traceback (last 3 lines):")
        for line in tb.strip().splitlines()[-3:]:
            print(f"      {line}")
        errors.append((nb_cell_idx, cell_idx+1, str(e), tb))

print()
print("=" * 60)
if errors:
    print(f"FAILED: {len(errors)} cell(s) raised errors:")
    for nb_idx, code_idx, msg, _ in errors:
        print(f"  Cell {nb_idx} (code #{code_idx}): {msg}")
    sys.exit(1)
else:
    print(f"SUCCESS: All {len(code_cells)} code cells executed without errors.")
    print("Notebook is clean end-to-end.")
