"""
Manuscript figure generation scripts.

This package contains scripts for generating all manuscript figures.
Each module can be run independently or through the manuscript_app CLI.

Subfolder layout maps to figure numbers:
  NMDetectiveAI/ → Fig2
  DMS/           → Fig3
  PE/            → Fig4
  LE/            → Fig5
  SP/            → Fig6
  selection/     → Fig7
  supplementary/ → Supplementary figures
"""

from . import (
    NMDetectiveAI,
    DMS,
    PE,
    LE,
    SPvar,
    selection,
    supplementary,
)

__all__ = [
    "NMDetectiveAI",
    "DMS",
    "PE",
    "LE",
    "SPvar",
    "selection",
    "supplementary",
]
