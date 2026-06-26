import base64
import hmac
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox
import logging

# Configure logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'license_check.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_secret_key():
    """Load the secret key from secure_key.dat."""
    try:
        if getattr(sys, '_MEIPASS', None):
            key_path = Path(sys._MEIPASS) / 'secure_key.dat'
            logging.debug(f"Looking for secure_key.dat in PyInstaller temp dir: {key_path}")
        else:
            key_path = Path(__file__).parent / 'secure_key.dat'
            logging.debug(f"Looking for secure_key.dat in script dir: {key_path}")

        if not key_path.exists():
            logging.warning("secure_key.dat not found, using default key.")
            return b"XYZ789ABC"

        with open(key_path, 'rb') as key_file:
            secret_key = key_file.read().strip()
            # Remove BOM if present
            if secret_key.startswith(b'\xef\xbb\xbf') or secret_key.startswith(b'\xff\xfe'):
                secret_key = secret_key[2:] if secret_key.startswith(b'\xff\xfe') else secret_key[3:]
            logging.debug(f"Raw key: {secret_key}")
            return secret_key
    except Exception as e:
        logging.error(f"Failed to load secure_key.dat: {e}")
        return b"XYZ789ABC"

SECRET_KEY = load_secret_key()

def validate_license_key(license_key):
    """Validate the license key against secure_key.dat."""
    try:
        logging.debug(f"Validating license key: {license_key}")
        payload_b64, signature = license_key.split(".")
        payload_b64_no_padding = payload_b64.rstrip("=")

        expected_signature = hmac.new(
            SECRET_KEY,
            payload_b64_no_padding.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        logging.debug(f"Provided signature: {signature}")
        logging.debug(f"Expected signature: {expected_signature}")

        if signature != expected_signature:
            logging.error("Invalid signature!")
            return False, "Invalid signature!"

        payload_json = base64.urlsafe_b64decode(payload_b64 + "==").decode("utf-8")
        payload = json.loads(payload_json)
        logging.debug(f"Decoded payload: {payload}")

        expiration_date = datetime.strptime(payload["expiration_date"], "%Y-%m-%d")
        start_date = datetime.strptime(payload["start_date"], "%Y-%m-%d")
        current_date = datetime.now()

        if start_date <= current_date <= expiration_date:
            logging.info("License key is valid.")
            return True, "License key is valid."
        else:
            logging.error("License expired or not yet valid!")
            return False, "License expired or not yet valid!"
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False, f"Validation error: {e}"

def prompt_license_key():
    """Prompt for license key using PyQt6 and validate it."""
    app = QApplication(sys.argv)
    try:
        # Load stored license key if exists
        license_file = Path(__file__).parent / 'license_data.dat'
        logging.debug(f"Checking for stored license at: {license_file}")
        if license_file.exists():
            # Use utf-8-sig here to strip BOM if present
            with open(license_file, 'r', encoding='utf-8-sig') as f:
                stored_key = f.read().strip()
                logging.debug(f"Loaded stored license key: {stored_key}")
                is_valid, message = validate_license_key(stored_key)
                if is_valid:
                    logging.debug("Stored license key validated successfully")
                    return True

        # Prompt for new license key
        key, ok = QInputDialog.getText(
            None,
            "License Key",
            "Enter your license key (contact salt19.llc@gmail.com to obtain one):",
            QLineEdit.EchoMode.Normal
        )
        if ok and key:
            is_valid, message = validate_license_key(key)
            if is_valid:
                with open(license_file, 'w', encoding='utf-8') as f:
                    f.write(key)
                logging.debug("New license key saved and validated")
                QMessageBox.information(None, "Success", message)
                return True
            else:
                QMessageBox.critical(None, "Error", message + " Exiting.")
                logging.error(f"License key validation failed: {message}")
                return False
        else:
            QMessageBox.critical(None, "Error", "No key entered. Exiting.")
            logging.warning("License prompt cancelled")
            return False
    except Exception as e:
        logging.error(f"Failed to process license prompt: {e}")
        QMessageBox.critical(None, "Error", f"License prompt error: {e}")
        return False
    finally:
        app.quit()

if __name__ == '__main__':
    sys.exit(0 if prompt_license_key() else 1)
