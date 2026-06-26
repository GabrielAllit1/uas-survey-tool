from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSettings, QMetaObject, QObject, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox,
    QHBoxLayout, QLabel, QLineEdit, QToolButton,
    QPushButton, QVBoxLayout,
)

__all__ = ["APIKeyManager"]

class _APIKeyDialog(QDialog):
    """Small dialog prompting for the OpenTopography API key."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent, flags=Qt.WindowType.Dialog)
        self.setWindowTitle("OpenTopography API Key")

        # Widgets
        lbl = QLabel("Paste your OpenTopography API key:")
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(320)
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)

        eye = QToolButton()
        eye.setIcon(QIcon.fromTheme("view-hidden"))
        eye.setCheckable(True)
        eye.toggled.connect(
            lambda v: (
                self.edit.setEchoMode(
                    QLineEdit.EchoMode.Normal if v else QLineEdit.EchoMode.Password
                ),
                eye.setIcon(QIcon.fromTheme("view-visible" if v else "view-hidden")),
            )
        )

        paste = QPushButton("Paste")
        paste.clicked.connect(self.edit.paste)

        self.chk_save = QCheckBox("Remember this key")
        self.chk_save.setChecked(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Layout
        hl = QHBoxLayout()
        hl.addWidget(self.edit, 1)
        hl.addWidget(eye)
        hl.addWidget(paste)

        vl = QVBoxLayout(self)
        vl.addWidget(lbl)
        vl.addLayout(hl)
        vl.addWidget(self.chk_save)
        vl.addWidget(buttons)

    @classmethod
    def get(cls, parent=None) -> tuple[Optional[str], bool]:
        dlg = cls(parent)
        ok = dlg.exec()
        return (dlg.edit.text() if ok == QDialog.DialogCode.Accepted else None,
                dlg.chk_save.isChecked())

class APIKeyManager(QObject):
    """Thread-safe singleton fetching/caching the OpenTopography key."""
    _SETTINGS_GROUP = "APIKeys"
    _KEY_NAME = "OpenTopography"
    _CONFIG_ENV_VAR = "UAS_SURVEY_TOOL_CONFIG"
    _cache: Optional[str] = None

    @classmethod
    def _config_dir(cls) -> Path:
        return Path.home() / ".uas_survey_tool"

    @classmethod
    def _candidate_config_paths(cls) -> list[Path]:
        configured = os.getenv(cls._CONFIG_ENV_VAR)
        paths: list[Path] = []
        if configured:
            paths.append(Path(configured))
        paths.append(Path.cwd() / "config.json")
        paths.append(cls._config_dir() / "config.json")
        return paths

    @classmethod
    def _read_config_file(cls) -> Optional[str]:
        for path in cls._candidate_config_paths():
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    key = data.get("opentopography_api_key")
                    if isinstance(key, str) and key.strip():
                        return key.strip()
            except Exception:
                continue
        return None

    @classmethod
    def _write_config_file(cls, key: str) -> None:
        target = cls._config_dir() / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "opentopography_api_key": key,
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def _read_settings(cls) -> Optional[str]:
        s = QSettings("UAS_Survey_Tool", "UAS_Survey_Tool")
        s.beginGroup(cls._SETTINGS_GROUP)
        key = s.value(cls._KEY_NAME, type=str)
        s.endGroup()
        return key

    @classmethod
    def _write_settings(cls, key: str) -> None:
        s = QSettings("UAS_Survey_Tool", "UAS_Survey_Tool")
        s.beginGroup(cls._SETTINGS_GROUP)
        s.setValue(cls._KEY_NAME, key)
        s.endGroup()

    @classmethod
    def opentopo(cls, parent=None) -> Optional[str]:
        """Return the cached OpenTopography key from env, config, settings, or prompt."""
        if cls._cache:
            return cls._cache

        # 1) Environment variable
        key = os.getenv("OPENTOPO_API_KEY")
        if key and key.strip():
            cls._cache = key.strip()
            return cls._cache

        # 2) Config file
        config_key = cls._read_config_file()
        if config_key:
            cls._cache = config_key
            return cls._cache

        # 3) Application settings
        stored = cls._read_settings()
        if stored:
            cls._cache = stored
            return cls._cache

        # 4) Prompt the user on first use
        app = QApplication.instance() or QApplication(sys.argv)
        if QThread.currentThread() is app.thread():
            key, save = _APIKeyDialog.get(parent)
        else:
            key_holder: list[str | None] = [None]
            save_holder: list[bool] = [False]

            def _run_dialog():
                k, s = _APIKeyDialog.get(parent)
                key_holder[0] = k
                save_holder[0] = s

            QMetaObject.invokeMethod(
                app, _run_dialog, Qt.ConnectionType.BlockingQueuedConnection
            )
            key, save = key_holder[0], save_holder[0]

        if key:
            key = key.strip()
        if key and save:
            cls._write_settings(key)
            cls._write_config_file(key)
        cls._cache = key if key else None
        return cls._cache
