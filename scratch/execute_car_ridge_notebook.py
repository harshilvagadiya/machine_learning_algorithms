import json
import uuid
import io
import sys
import os
import base64
from pathlib import Path

workspace_root = Path("/home/python/03_Custom_addons/ML")
nb_path = workspace_root / "supervised_learning/RidgeRegression/Car_Price_Prediction_Ridge.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells to execute: {len(nb['cells'])}")

# Change working directory to notebook's dir
os.chdir(workspace_root / "supervised_learning/RidgeRegression")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

namespace = {}
captured_figures = []

def custom_show(*args, **kwargs):
    fig = plt.gcf()
    if fig.get_axes():  # If there is content in the figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        captured_figures.append(img_base64)
    plt.close("all")

namespace["plt"] = plt
plt.show = custom_show

exec_count = 1
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    code = "".join(cell["source"])
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    captured_figures.clear()
    
    print(f"Executing Cell {idx}...", file=sys.stderr)
    try:
        exec(code, namespace)
        output_text = buffer.getvalue()
        cell["outputs"] = []
        
        if output_text:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [l + "\n" for l in output_text.splitlines()]
            })
            
        for img in captured_figures:
            cell["outputs"].append({
                "data": {
                    "image/png": img,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            
        cell["execution_count"] = exec_count
        exec_count += 1
    except Exception as e:
        sys.stdout = old_stdout
        print(f"❌ Error in cell {idx}: {e}", file=sys.stderr)
        raise e
    finally:
        sys.stdout = old_stdout

os.chdir(workspace_root)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ All {len(nb['cells'])} cells successfully executed and saved with outputs and plots!")

