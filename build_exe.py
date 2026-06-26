import os
import subprocess
from shutil import which

def build_exe():
    os.system("pyinstaller UAS_Survey_Tool_v2.spec")

def sign_exe():
    signtool = which("signtool")
    exe_path = "dist/UAS_Survey_Tool_v2/UAS_Survey_Tool_v2.exe"
    if not signtool:
        print("❌ signtool.exe not found in PATH.")
        return
    cmd = f'{signtool} sign /a /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /v "{exe_path}"'
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    build_exe()
    # Uncomment to sign:
    # sign_exe()
