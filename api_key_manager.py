from __future__ import annotations

import sys
import os
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
    _cache: Optional[str] = None
    _DEFAULT_API_KEY = "e1d1687702fdc48744f29e0d52503e3f"  # Provided API key

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
        """Return the cached key or use the default/provided key."""
        if cls._cache:
            return cls._cache

        # 1) Check environment variable
        key = os.getenv("OPENTOPO_API_KEY")
        if key:
            cls._cache = key
            return cls._cache

        # 2) Check settings
        stored = cls._read_settings()
        if stored:
            cls._cache = stored
            return cls._cache

        # 3) Use provided default key
        cls._cache = cls._DEFAULT_API_KEY
        cls._write_settings(cls._cache)
        return cls._cache

        # 4) Fallback to GUI dialog (only if needed)
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

        if key and save:
            cls._write_settings(key)
        cls._cache = key
        return cls._cache