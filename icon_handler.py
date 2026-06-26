import os
from PyQt6.QtGui import QIcon

def get_app_icon(layout_mode=None):
    """
    Returns the application icon based on layout mode.
    Falls back to default icon if specific icon is missing.
    """
    try:
        base_path = os.path.dirname(__file__)
        icon_map = {
            "fractal": "fractal_icon.ico",
            "fibonacci": "fibonacci_icon.ico",
            None: "icon.ico"
        }
        icon_file = icon_map.get(layout_mode, "icon.ico")
        icon_path = os.path.join(base_path, icon_file)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        else:
            print(f"[WARNING] Icon file {icon_file} not found. Using default icon.")
            default_path = os.path.join(base_path, "icon.ico")
            if os.path.exists(default_path):
                return QIcon(default_path)
            return QIcon()  # Empty fallback icon
    except Exception as e:
        print(f"[ERROR] Icon loading failed: {e}")
        return QIcon()