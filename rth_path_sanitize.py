# rth_path_sanitize.py
import os
bad_tokens = ["arbor_analyzer"]
parts = os.environ.get("PATH", "").split(";")
parts = [p for p in parts if p and all(bt.lower() not in p.lower() for bt in bad_tokens)]
# Keep order but de‑dup
seen, clean = set(), []
for p in parts:
    if p not in seen:
        clean.append(p); seen.add(p)
os.environ["PATH"] = ";".join(clean)
