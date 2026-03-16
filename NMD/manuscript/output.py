"""
Output path resolution for manuscript figure scripts.

Provides a unified interface for resolving output paths in two modes:
1. **Standalone mode** (`python -m NMD.manuscript.<folder>.<script>`):
   - Figures → FIGURES_DIR / "manuscript" / "<script_name>.png"
   - Tables → TABLES_DIR / "manuscript" / "<script_name>.csv"

2. **Manuscript app mode** (via manuscript CLI):
   - Figures → MANUSCRIPT_FIGURES_DIR / "<FigN>" / "<FigN><panel>.png"
   - Source data → MANUSCRIPT_FIGURES_DIR / "source_data" / "<FigN>" / "<FigN><panel>.csv"
   - Shared tables → MANUSCRIPT_TABLES_DIR / "<name>.csv" (important/shared)
                   or TABLES_DIR / "analysis" / "<name>.csv" (internal)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from NMD.config import (
    FIGURES_DIR,
    TABLES_DIR,
    MANUSCRIPT_FIGURES_DIR,
    MANUSCRIPT_TABLES_DIR,
)


@dataclass
class ManuscriptPaths:
    """Resolved output paths for a manuscript figure script.

    Attributes:
        figure_png: Path for the PNG figure output.
        figure_pdf: Path for the PDF figure output.
        source_data: Path for the source data file (CSV or Excel).
        script_name: Name of the script (without extension), used for
            default file naming in standalone mode.
        figure_label: Figure label (e.g. "Fig2a") when run from app.
        figure_number: Figure number string (e.g. "Fig2") when run from app.
    """

    figure_png: Path
    figure_pdf: Path
    source_data: Path
    script_name: str
    figure_label: Optional[str] = None
    figure_number: Optional[str] = None

    @property
    def is_manuscript_mode(self) -> bool:
        """True when running through the manuscript app."""
        return self.figure_label is not None


def get_paths(
    script_name: str,
    figure_label: Optional[str] = None,
    figure_number: Optional[str] = None,
    source_data_ext: str = ".csv",
) -> ManuscriptPaths:
    """Resolve output paths for a manuscript script.

    Args:
        script_name: Module/script name without extension (e.g.
            "NMDetective_training_curves").  Used as the default file stem
            in standalone mode.
        figure_label: Panel label like "Fig2a" — set by the manuscript app.
            When *None*, standalone mode paths are returned.
        figure_number: Figure number like "Fig2" — set by the manuscript app.
        source_data_ext: Extension for the source-data file (".csv" or
            ".xlsx").  Defaults to ".csv".

    Returns:
        A :class:`ManuscriptPaths` instance with all output paths resolved
        and parent directories created.
    """
    if figure_label is not None:
        # --- Manuscript app mode ---
        if figure_number is None:
            raise ValueError("figure_number is required when figure_label is set")

        fig_dir = MANUSCRIPT_FIGURES_DIR / figure_number
        source_dir = MANUSCRIPT_FIGURES_DIR / "source_data" / figure_number

        paths = ManuscriptPaths(
            figure_png=fig_dir / f"{figure_label}.png",
            figure_pdf=fig_dir / f"{figure_label}.pdf",
            source_data=source_dir / f"{figure_label}{source_data_ext}",
            script_name=script_name,
            figure_label=figure_label,
            figure_number=figure_number,
        )
    else:
        # --- Standalone mode ---
        fig_dir = FIGURES_DIR / "manuscript"
        table_dir = TABLES_DIR / "manuscript"

        paths = ManuscriptPaths(
            figure_png=fig_dir / f"{script_name}.png",
            figure_pdf=fig_dir / f"{script_name}.pdf",
            source_data=table_dir / f"{script_name}{source_data_ext}",
            script_name=script_name,
        )

    # Ensure parent directories exist
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    paths.source_data.parent.mkdir(parents=True, exist_ok=True)

    return paths


def get_analysis_table_path(name: str) -> Path:
    """Return a path under TABLES_DIR / 'analysis' for internal/shared tables.

    These are intermediate analysis results that may be reused across scripts
    but are not part of the manuscript source data.

    Args:
        name: File name (with extension), e.g. "long_exon_pca_matrix.csv".

    Returns:
        Absolute path, with parent directory created.
    """
    path = TABLES_DIR / "analysis" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_manuscript_table_path(name: str) -> Path:
    """Return a path under MANUSCRIPT_TABLES_DIR for important shared tables.

    These are supplementary tables intended for publication (e.g. PCA results,
    model parameters).

    Args:
        name: File name (with extension), e.g. "penultimate_exon_fits.csv".

    Returns:
        Absolute path, with parent directory created.
    """
    path = MANUSCRIPT_TABLES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
