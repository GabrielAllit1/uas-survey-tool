# build_and_sign_windows.py  (drop-in replacement)
import os
import sys
import subprocess
import shutil
import argparse
import datetime
import zipfile
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_command(command, check=True):
    logging.info("Executing: %s", " ".join(map(str, command)))
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=check)
        if result.stdout:
            logging.info(result.stdout.strip())
        if result.stderr:
            logging.warning(result.stderr.strip())
        return result
    except subprocess.CalledProcessError as e:
        logging.error("❌ Command failed.")
        logging.error("STDOUT:\n%s", e.stdout)
        logging.error("STDERR:\n%s", e.stderr)
        raise
    except FileNotFoundError:
        logging.error("❌ Command not found: %s", command[0])
        raise

def check_prerequisites():
    for tool in ["python", "pyinstaller"]:
        run_command([tool, "--version"])

def clean_build(project_dir: Path):
    for folder in ["dist", "build"]:
        p = project_dir / folder
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            logging.info("Removed previous build folder: %s", p)

def build_exe(project_dir: Path, spec_file="UAS_Survey_Tool_v2.spec"):
    logging.info("📦 Building executable with PyInstaller...")
    run_command([sys.executable, "-m", "PyInstaller", str(project_dir / spec_file), "--clean", "--noconfirm"])

def find_exe(dist_dir: Path, preferred_name: str | None = None) -> Path | None:
    # Try preferred name first (onefile)
    if preferred_name:
        candidate = dist_dir / preferred_name
        if candidate.exists():
            return candidate

    # Onefile exe at top-level dist/
    exes = list(dist_dir.glob("*.exe"))
    if exes:
        # Pick newest
        return max(exes, key=lambda p: p.stat().st_mtime)

    # Onedir: dist/Name/Name.exe
    for sub in dist_dir.iterdir():
        if sub.is_dir():
            exe = sub / (sub.name + ".exe")
            if exe.exists():
                return exe
    return None

def sign_exe(exe_path: Path, cert_thumbprint: str | None):
    if not cert_thumbprint:
        logging.info("Skipping code signing (no certificate thumbprint provided).")
        return
    logging.info("🔏 Signing executable: %s", exe_path)
    run_command(["signtool", "sign", "/sha1", cert_thumbprint, "/tr", "http://timestamp.digicert.com", "/td", "SHA256", "/v", str(exe_path)])

def compute_metadata(exe_path: Path):
    with open(exe_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    size_mb = round(exe_path.stat().st_size / (1024 * 1024), 2)
    return {
        "exe_name": exe_path.name,
        "sha256": sha256,
        "size_mb": size_mb,
        "build_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def create_readme(dist_dir: Path, exe_name: str, metadata: dict) -> Path:
    readme_path = dist_dir / "README.txt"
    readme_content = f"""UAS Survey Tool v2 - {exe_name}

Build Date: {metadata['build_date']}
SHA256: {metadata['sha256']}
Size: {metadata['size_mb']} MB

Requirements:
- Windows 10 or 11 (64-bit)
- No Python, GDAL, or dependencies required

Usage:
1. Double-click {exe_name} to start
2. Load your KMZ and generate outputs
3. Deliverables include PDF, CSV, KML
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    return readme_path

def create_zip(exe_path: Path, dist_dir: Path, metadata: dict, readme_path: Path) -> Path:
    zip_path = dist_dir / f"{exe_path.stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(exe_path, exe_path.name)
        # Write metadata
        meta_path = dist_dir / f"{exe_path.stem}_metadata.txt"
        meta_path.write_text("\n".join(f"{k}: {v}" for k, v in metadata.items()), encoding="utf-8")
        zipf.write(meta_path, meta_path.name)
        zipf.write(readme_path, readme_path.name)
    logging.info("📦 Created ZIP archive: %s", zip_path)
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="Build, sign, and package UAS Survey Tool")
    parser.add_argument("--project-dir", default=r"C:\Users\gabri\Documents\UAS_Survey_Tool_v2.0")
    parser.add_argument("--spec-file", default="UAS_Survey_Tool_v2.spec")
    parser.add_argument("--exe-name", help="Optional exe name to pick in dist/")
    parser.add_argument("--cert-thumbprint", help="Optional SHA1 thumbprint for code signing")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    dist_dir = project_dir / "dist"

    check_prerequisites()
    clean_build(project_dir)
    build_exe(project_dir, args.spec_file)

    exe_path = find_exe(dist_dir, args.exe_name)
    if not exe_path:
        logging.error("❌ Could not locate built exe in %s", dist_dir)
        sys.exit(2)

    sign_exe(exe_path, args.cert_thumbprint)
    metadata = compute_metadata(exe_path)
    readme = create_readme(dist_dir, exe_path.name, metadata)
    zip_path = create_zip(exe_path, dist_dir, metadata, readme)

    # Optional copy to release folder
    release_dir = project_dir / "dist_release"
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe_path, release_dir / exe_path.name)
    logging.info("✅ Copied to release: %s", release_dir / exe_path.name)
    logging.info("✅ All done! Distribution package ready at: %s", zip_path)

if __name__ == "__main__":
    main()
