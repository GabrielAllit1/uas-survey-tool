import base64
import hmac
import hashlib
import json
import os

def load_secret_key():
    key_path = os.path.join(os.path.dirname(__file__), 'secure_key.dat')
    if not os.path.exists(key_path):
        print("secure_key.dat not found, using default key.")
        return b"XYZ789ABC"
    with open(key_path, 'rb') as key_file:
        return key_file.read().strip()  # Remove extra spaces or newlines

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