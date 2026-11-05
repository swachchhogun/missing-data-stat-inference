"""Write simulation_core.py from the notebook's own code cells, so the report uses exactly the notebook's functions."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
nb = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/notebook/missing_data_simulation.ipynb"))
WANT = ["1a", "1b", "2a", "3a", "4a", "4b", "4c", "4d", "5.2a", "5.3a", "5.4a", "5.5a", "5.5b", "5.5c", "5.5d",
        "5.6a", "5.7a", "5.8a", "5.8b", "5.8c", "5.8d", "5.9a", "5.9b", "6a", "6b"]
cells = {}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    s = "".join(c["source"]); m = re.match(r"# ===== Step ([\w\.]+) ", s)
    if m: cells[m.group(1)] = s
out = ['"""Generated from the notebook by extract_core.py - do not edit by hand."""']
for k in WANT:
    s = cells[k]
    if k == "1a": s = s.replace("from IPython.display import display, Markdown", "")
    if k == "1b":   # keep only the constants, not the folder handling
        s = "\n".join(l for l in s.split("\n") if not any(w in l for w in ("RESULTS_DIR", "print(", "from pathlib")))
    out.append(s)
open(f"{HERE}/simulation_core.py", "w").write("\n\n".join(out) + "\n")
print("simulation_core.py written with", len(WANT), "notebook steps")
