# bootstrap_main.py
from __future__ import annotations
import sys
import logging
import importlib
from PyQt6.QtWidgets import QApplication

import proj_env  # ensure PROJ/GDAL env before anything else

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("bootstrap")

ENTRY_CLASSES = ("UASWindow", "MainWindow", "ModernUI", "UASSurveyMainWindow", "AppWindow")
ENTRY_FUNCS = ("main", "run", "start", "launch", "launch_ui")

def main() -> int:
    LOGGER.info("Starting UAS Survey Tool…")
    app = QApplication.instance() or QApplication(sys.argv)

    ui = importlib.import_module("modern_ui")

    # (a) prefer a callable entry point if present
    for fn in ENTRY_FUNCS:
        if hasattr(ui, fn) and callable(getattr(ui, fn)):
            LOGGER.info("Invoking modern_ui.%s()", fn)
            return int(getattr(ui, fn)())

    # (b) otherwise look for a window class
    for cls_name in ENTRY_CLASSES:
        if hasattr(ui, cls_name):
            LOGGER.info("Instantiating modern_ui.%s", cls_name)
            window_cls = getattr(ui, cls_name)
            win = window_cls()
            win.show()
            return app.exec()

    found_classes = [name for name in dir(ui) if name[0].isupper()]
    found_funcs = [name for name in dir(ui) if callable(getattr(ui, name))]
    LOGGER.error(
        "modern_ui has no callable entry point %s and no known window class %s.\n"
        "Found classes: %s\nFound functions: %s",
        ENTRY_FUNCS, ENTRY_CLASSES, found_classes, found_funcs
    )
    return 1

if __name__ == "__main__":
    sys.exit(main())
