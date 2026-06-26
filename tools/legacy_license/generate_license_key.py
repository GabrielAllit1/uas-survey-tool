import base64
import hmac
import hashlib
import json
import os
from pathlib import Path

def load_secret_key():
    env_secret = os.getenv("UAS_LICENSE_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")

    configured_key_path = os.getenv("UAS_LICENSE_SECRET_FILE")
    candidates = []
    if configured_key_path:
        candidates.append(Path(configured_key_path))
    candidates.append(Path(__file__).parent / 'secure_key.dat')

    for key_path in candidates:
        if key_path.exists():
            with open(key_path, 'rb') as key_file:
                return key_file.read().strip()
    raise RuntimeError(
        "No license signing secret found. "
        "Set UAS_LICENSE_SECRET or UAS_LICENSE_SECRET_FILE before generating keys."
    )

SECRET_KEY = load_secret_key()

# Example dates
start_date = "2025-05-01"
expiration_date = "2026-06-30"

# Create payload without user_id
payload = {
    "start_date": start_date,
    "expiration_date": expiration_date
}
payload_json = json.dumps(payload, sort_keys=True)  # Ensure consistent ordering
payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
payload_b64_no_padding = payload_b64.rstrip("=")  # Remove padding

# Generate signature
signature = hmac.new(
    SECRET_KEY,
    payload_b64_no_padding.encode("utf-8"),
    hashlib.sha256
).hexdigest()

# Combine payload and signature into license key
license_key = f"{payload_b64}.{signature}"
print(f"Generated license key: {license_key}")
