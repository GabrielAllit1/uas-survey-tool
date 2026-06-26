import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'geometry_type_checker.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def find_to_string_calls(root_dir):
    """Search for .to_string() calls in Python files."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if ".to_string(" in line:
                                logging.debug(f"Found '.to_string()' in {file_path} at line {lineno}:")
                                logging.debug(f"    {line.strip()}")
                except UnicodeDecodeError:
                    logging.error(f"Failed to read {file_path}: Non-UTF-8 encoding")
                except Exception as e:
                    logging.error(f"Failed to read {file_path}: {e}")

if __name__ == "__main__":
    project_root = r"C:\Users\gabri\Documents\UAS_Survey_Tool_v2.0"
    find_to_string_calls(project_root)