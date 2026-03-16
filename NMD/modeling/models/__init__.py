"""
NMD prediction models.

This module provides three model implementations:
- NMDetectiveA: Random Forest Regressor (classical ML)
- NMDetectiveB: Decision Tree with fixed rules (classical ML)
- NMDetectiveAI: Deep learning model with Orthrus encoder (deep learning)
"""

from .NMDetectiveA import NMDetectiveA
from .NMDetectiveB import NMDetectiveB
from .NMDetectiveAI import NMDetectiveAI

__all__ = [
    "NMDetectiveA",
    "NMDetectiveB",
    "NMDetectiveAI",
]
