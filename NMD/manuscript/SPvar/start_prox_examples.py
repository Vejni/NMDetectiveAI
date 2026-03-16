"""
Start-proximal gene examples with sigmoid fits (Fig 6b).

Plots sigmoid fits for selected genes overlaid on a genomewide PTC LOESS
curve to illustrate gene-level variation in start-proximal NMD evasion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from loguru import logger

from NMD.config import COLOURS, PROCESSED_DATA_DIR
from NMD.manuscript.output import get_paths
from NMD.analysis.dms_sigmoid_fitting import fit_logistic, logistic4

SCRIPT_NAME = "start_prox_examples"

# Input paths
DMS_SP_FILE = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
PTC_FILE = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.csv"

# Filtering
PTC_POS_MAX = 250

# Genes to highlight
HIGHLIGHT_GENES = ["EPHA5", "NF2", "KDM6A", "TP63"]
GENE_COLORS = ["#4daf4a", "#ff7f00", "#a65628", "#f781bf"]

FIGURE_SIZE = (12, 8)
DPI = 300


def _process_ptc_data() -> pd.DataFrame:
    """Load and filter PTC TCGA data for the start-proximal region."""
    ptc_df = pd.read_csv(PTC_FILE)
    ptc_df = ptc_df[
        ~ptc_df.Last_Exon
        & ~ptc_df.Penultimate_Exon
        & ~ptc_df.Long_Exon
        & (ptc_df.PTC_CDS_pos <= PTC_POS_MAX)
        & (ptc_df.Ref != "-")
        & (ptc_df.Alt != "-")
    ]
    logger.info(f"Filtered PTC data: {len(ptc_df)} variants")
    return ptc_df


def _build_source_data(dms_data: pd.DataFrame, ptc_data: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy DataFrame containing only the data shown on the plot."""
    rows = []
    # DMS observations for highlighted genes
    for gene in HIGHLIGHT_GENES:
        gene_df = dms_data[dms_data["gene"] == gene].dropna(subset=["PTCposition_nt", "NMDeff_Norm"])
        for _, r in gene_df.iterrows():
            rows.append({
                "data_type": "DMS",
                "gene": gene,
                "position": r["PTCposition_nt"],
                "NMDeff_Norm": r["NMDeff_Norm"],
            })
    # PTC data (for LOESS fit)
    ptc_clean = ptc_data.dropna(subset=["PTC_CDS_pos", "NMDeff_Norm"])
    for _, r in ptc_clean.iterrows():
        rows.append({
            "data_type": "PTC",
            "gene": np.nan,
            "position": r["PTC_CDS_pos"],
            "NMDeff_Norm": r["NMDeff_Norm"],
        })
    return pd.DataFrame(rows)


def plot_normalization_comparison(dms_data: pd.DataFrame, ptc_data: pd.DataFrame):
    """Create plot showing sigmoid fits for highlighted genes vs PTC LOESS."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # PTC LOESS
    ptc_clean = ptc_data.dropna(subset=["PTC_CDS_pos", "NMDeff_Norm"])
    ptc_smooth = lowess(ptc_clean["NMDeff_Norm"], ptc_clean["PTC_CDS_pos"], frac=0.4)
    ax.plot(
        ptc_smooth[:, 0], ptc_smooth[:, 1],
        color="gray", linewidth=3, linestyle="--",
        label=f"TCGA genomewide PTCs LOESS fit (n={len(ptc_clean)})", alpha=0.8, zorder=1,
    )

    # Overall DMS sigmoid
    dms_clean = dms_data.dropna(subset=["PTCposition_nt", "NMDeff_Norm"])
    try:
        all_result = fit_logistic(dms_clean["PTCposition_nt"].values, dms_clean["NMDeff_Norm"].values)
        x_smooth = np.linspace(dms_clean["PTCposition_nt"].min(), dms_clean["PTCposition_nt"].max(), 300)
        x_scaled = (x_smooth - all_result["x_min"]) / (all_result["x_max"] - all_result["x_min"])
        y_sigmoid = logistic4(x_scaled, *all_result["params"])
        ax.plot(
            x_smooth, y_sigmoid, color=COLOURS[1], linewidth=3, linestyle=":",
            label=f'Average DMS sigmoid (R\u00b2={all_result["r2"]:.2f})', alpha=0.8, zorder=2,
        )
    except Exception as e:
        logger.warning(f"Failed to fit sigmoid to all DMS data: {e}")

    # Per-gene sigmoid fits
    for i, gene in enumerate(HIGHLIGHT_GENES):
        gene_clean = dms_data[dms_data["gene"] == gene].dropna(subset=["PTCposition_nt", "NMDeff_Norm"])
        if len(gene_clean) <= 10:
            continue
        ax.scatter(
            gene_clean["PTCposition_nt"], gene_clean["NMDeff_Norm"],
            color=GENE_COLORS[i], alpha=0.3, s=20, label=f"{gene} observations", zorder=2,
        )
        try:
            result = fit_logistic(gene_clean["PTCposition_nt"].values, gene_clean["NMDeff_Norm"].values)
            xg = np.linspace(gene_clean["PTCposition_nt"].min(), gene_clean["PTCposition_nt"].max(), 200)
            xg_s = (xg - result["x_min"]) / (result["x_max"] - result["x_min"])
            ax.plot(
                xg, logistic4(xg_s, *result["params"]),
                color=GENE_COLORS[i], linewidth=3,
                label=f'{gene} sigmoid (R\u00b2={result["r2"]:.2f})', zorder=3,
            )
        except Exception as e:
            logger.warning(f"Failed to fit sigmoid for {gene}: {e}")

    ax.set_xlabel("PTC Position (nt)", fontsize=18, fontweight="bold")
    ax.set_ylabel("NMD efficiency", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12, loc="best", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    return fig


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate start-proximal examples figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, load it instead of recomputing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )

    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        source = pd.read_csv(paths.source_data)
        dms_data = source[source["data_type"] == "DMS"].rename(columns={"position": "PTCposition_nt"})
        ptc_data = source[source["data_type"] == "PTC"].rename(columns={"position": "PTC_CDS_pos"})
    else:
        dms_data = pd.read_csv(DMS_SP_FILE)
        ptc_data = _process_ptc_data()
        # Save only the data that appears on the plot
        source = _build_source_data(dms_data, ptc_data)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_normalization_comparison(dms_data, ptc_data)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Start-proximal examples complete!")


if __name__ == "__main__":
    main()
