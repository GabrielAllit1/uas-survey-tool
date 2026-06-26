import glob
import re
import shutil

pattern = re.compile(r'f[\'"]\s*\{lon\},\s*\{lat\}[\'"]')
replacement = 'f"{lat},{lon}"'

for filename in glob.glob("**/*.py", recursive=True):
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    for line in lines:
        if pattern.search(line):
            new_line = pattern.sub(replacement, line)
            if new_line != line:
                print(f"Fixing {filename}: {line.strip()} -> {new_line.strip()}")
                modified = True
                new_lines.append(new_line)
                continue
        new_lines.append(line)

    if modified:
        # Backup original
        shutil.copy2(filename, filename + ".bak")
        # Overwrite file with fixed lines
        with open(filename, "w", encoding="utf8") as f:
            f.writelines(new_lines)
