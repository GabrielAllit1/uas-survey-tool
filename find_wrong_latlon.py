import glob
import re

for f in glob.glob("**/*.py", recursive=True):
    with open(f, encoding="utf8", errors="ignore") as file:
        for i, l in enumerate(file):
            if re.search(r'f[\"\\\']\s*{lon},\s*{lat}[\"\\\']', l):
                print(f"{f}:{i+1}: {l.strip()}")
