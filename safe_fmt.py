# safe_fmt.py
from __future__ import annotations
from typing import Any

def f2(v: Any, default: str = "n/a") -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return default

def f1(v: Any, default: str = "n/a") -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return default

def fi(v: Any, default: str = "n/a") -> str:
    try:
        return f"{int(round(float(v)))}"
    except Exception:
        return default

def s(v: Any, default: str = "n/a") -> str:
    return default if v is None else str(v)
