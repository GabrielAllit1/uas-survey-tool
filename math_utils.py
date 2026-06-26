"""
math_utils.py ─ UAS Survey Tool v2.0
────────────────────────────────────────────────────────────────────────────
Numerical helpers used across the application.
"""

from __future__ import annotations
import math
import numpy as np
from scipy import signal
import logging
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────
_log_dir = Path(__file__).with_name("logs")
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_dir / "uas_survey_tool.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
#  Fibonacci utilities
# ──────────────────────────────────────────────────────────────────────────
def fibonacci(n: int) -> list[int]:
    """
    Return the first *n* Fibonacci numbers (1-indexed, starting with 1, 1).

    Raises
    ------
    ValueError
        If *n* ≤ 0.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    _log.debug("fibonacci(%d) → %s … len=%d", n, seq[:10], len(seq))
    return seq[:n]

# ──────────────────────────────────────────────────────────────────────────
#  Unified pattern / anomaly detector
# ──────────────────────────────────────────────────────────────────────────
def unified_detection(
    index: int,
    phase: float = 0.0,
    *,
    weights: tuple[float, float, float, float] | list[float] = (0.25, 0.25, 0.25, 0.25),
    clip: tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Blend four independent kernels into a single **pattern score** ∈ [clip[0], clip[1]].

    Kernels
    -------
    1. *Sinusoidal* – cyclical regularity  
    2. *Logistic-map* – bounded chaos  
    3. *Golden-angle rotation* – low-discrepancy fill  
    4. *Fibonacci mod prime* – pseudo-random jitter

    Parameters
    ----------
    index
        0-based point index.
    phase
        Optional phase shift for the sinusoidal component.
    weights
        Relative weights for the four kernels; any non-negative numbers.
    clip
        Output range.

    Returns
    -------
    float
        Deterministic score in the requested range.
    """
    if len(weights) != 4 or any(w < 0 for w in weights):
        raise ValueError("weights must be four non-negative numbers")

    # normalise weights
    s = float(sum(weights))
    w1, w2, w3, w4 = (w / s for w in weights)

    # 1 ─ sinusoidal  ∈ [0,1]
    k1 = 0.5 * (1.0 + math.sin(index + phase))

    # 2 ─ logistic map chaos  ∈ (0,1)
    r   = 3.99
    x   = 0.5
    for _ in range((index % 12) + 1):          # ≤12 iterations keeps it O(1)
        x = r * x * (1.0 - x)
    k2 = x

    # 3 ─ golden-angle (φ) rotation  ∈ [0,1)
    phi = (math.sqrt(5) - 1.0) / 2.0
    k3  = (index * phi) % 1.0

    # 4 ─ Fibonacci mod prime  ∈ [0,1)
    prime = 1_000_003                            # ≈1 million, prime
    if index < 2:
        fib_mod = 1
    else:
        a, b = 1, 1
        for _ in range(2, index + 1):
            a, b = b, (a + b) % prime
        fib_mod = b
    k4 = fib_mod / prime

    # weighted sum
    raw = w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4

    lo, hi = clip
    score  = max(lo, min(hi, raw))
    _log.debug(
        "unified_detection(idx=%d) → %.4f  [k1=%.3f k2=%.3f k3=%.3f k4=%.3f]",
        index,
        score,
        k1,
        k2,
        k3,
        k4,
    )
    return score

# ──────────────────────────────────────────────────────────────────────────
#  Elevation cycle detector
# ──────────────────────────────────────────────────────────────────────────
def detect_elevation_cycles(elevations: list[float]) -> float:
    """
    Detect cyclic structure in a 1-D elevation profile via autocorrelation.

    Returns
    -------
    float
        Normalised cycle score ∈ [0, 1].
    """
    if len(elevations) < 3:
        _log.warning("detect_elevation_cycles: <3 samples")
        return 0.0

    elev = np.asarray([e for e in elevations if e is not None and not np.isnan(e)])
    if elev.size < 3:
        _log.warning("detect_elevation_cycles: no valid samples")
        return 0.0

    elev_norm = (elev - elev.mean()) / (elev.std() + 1e-9)
    ac = signal.correlate(elev_norm, elev_norm, mode="full")[elev_norm.size - 1 :]
    peaks, _ = signal.find_peaks(ac, height=0.2)          # require modest peak

    score = len(peaks) / elev_norm.size
    _log.debug("detect_elevation_cycles → peaks=%d, score=%.4f", len(peaks), score)
    return score
