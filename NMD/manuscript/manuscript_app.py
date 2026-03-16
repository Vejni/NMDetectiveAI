"""
Manuscript figures CLI app.

Provides CLI commands to generate all manuscript figures.  Each command
wraps the corresponding script in the NMD/manuscript/ sub-packages.

Two running modes
-----------------
* **Standalone** (``python -m NMD.manuscript.<folder>.<script>``):
  Figures → ``FIGURES_DIR/manuscript/<script>.png``
  Tables  → ``TABLES_DIR/manuscript/<script>.csv``

* **Manuscript app** (``python -m main manuscript <command>``):
  Figures     → ``MANUSCRIPT_FIGURES_DIR/<FigN>/<FigN><panel>.png``
  Source data → ``MANUSCRIPT_FIGURES_DIR/source_data/<FigN><panel>.csv``
"""

import typer
from loguru import logger

# ── Imports of every manuscript script module ─────────────────────────────
from NMD.manuscript.NMDetectiveAI import (
    NMDetective_training_curves,
    NMDetective_comparison,
    data_interreplicate_correlation,
    NMDetective_recurrent_PTCs,
)
from NMD.manuscript.DMS import DMS_interreplicate_correlations
from NMD.manuscript.PE import (
    penultimate_exon_predictions,
    penultimate_exon_fit_quality,
    penultimate_exon_curves,
)
from NMD.manuscript.LE import (
    long_exon_overview,
    long_exon_regression,
    long_exon_pca_scatter,
    long_exon_pca_curves,
    long_exon_pca_correlations,
    long_exon_pca_examples,
)
from NMD.manuscript.SPreinit import (
    reinit_boxplot,
    reinit_intercistronic,
    reinit_kozak,
    reinit_qpcr_validation,
)
from NMD.manuscript.context import (
    context_hexamers,
    context_upstream_codon,
    context_ptc_position,
)
from NMD.manuscript.SPvar import (
    start_prox_pca,
    start_prox_examples,
    start_prox_correlations,
    start_prox_predictions,
    start_prox_comparison,
    start_prox_clustering,
)
from NMD.manuscript.selection import (
    gnomad_disease_genes,
    gnomad_disease_gene_enrichment,
    gnomad_prediction_distributions,
    gnomad_nmd_pie_charts,
    gnomad_nmd_bar_chart,
    gnomad_rarity_and_constrain,
    TCGA_TSG_OG,
    TCGA_cancer_genes,
)
from NMD.manuscript.supplementary import (
    NMDeff_distributions,
    cancer_gene_selection_scores,
    preprocessing_stats,
    top_long_exons,
    hyperparameter_sweep,
    long_exon_DMS_PTC_comparison_full,
    penultimate_exon_genome_wide,
    SPreinit_ptc_curves,
    SPreinit_ptc_curves_aug,
    context_downstream_trinuc,
    gnomad_logodds_correlation,
    DMS_interreplicate_correlations_full,
)

from NMD.manuscript.output import get_paths

app = typer.Typer(help="Generate manuscript figures")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE → PANEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: (panel_letters, module, source_data_ext)
#   panel_letters: str of panel letter(s), e.g. "a" or "cde" for multi-panel
#   source_data_ext: ".csv" for single panel, ".xlsx" for multi-panel/sheet

FIGURE_REGISTRY: dict[str, list[tuple[str, object, str]]] = {
    "Fig2": [
        ("a", NMDetective_training_curves, ".csv"),
        ("b", NMDetective_comparison, ".csv"),
        ("cde", data_interreplicate_correlation, ".xlsx"),
        ("f", NMDetective_recurrent_PTCs, ".csv"),
    ],
    "Fig3": [
        ("cde", DMS_interreplicate_correlations, ".xlsx"),
    ],
    "Fig4": [
        ("bc", penultimate_exon_predictions, ".xlsx"),
        ("d", penultimate_exon_fit_quality, ".xlsx"),
        ("e", penultimate_exon_curves, ".xlsx"),
    ],
    "Fig5": [
        ("b", long_exon_overview, ".xlsx"),
        ("c", long_exon_regression, ".csv"),
        ("d", long_exon_pca_scatter, ".csv"),
        ("e", long_exon_pca_curves, ".csv"),
        ("f", long_exon_pca_correlations, ".csv"),
        ("g", long_exon_pca_examples, ".xlsx"),
    ],
    "Fig6": [
        ("b", start_prox_examples, ".csv"),
        ("c", start_prox_pca, ".csv"),
        ("d", start_prox_correlations, ".csv"),
        ("e", start_prox_predictions, ".csv"),
        ("f", start_prox_comparison, ".csv"),
        ("g", start_prox_clustering, ".csv"),
    ],
    "Fig7": [
        ("b", reinit_boxplot, ".csv"),
        ("c", reinit_intercistronic, ".csv"),
        ("d", reinit_kozak, ".csv"),
        ("e", reinit_qpcr_validation, ".csv"),
        ("g", context_upstream_codon, ".csv"),
        ("h", context_hexamers, ".csv"),
        ("i", context_ptc_position, ".csv"),
    ],
    "Fig8": [
        ("a", gnomad_prediction_distributions, ".csv"),
        ("b", gnomad_nmd_pie_charts, ".xlsx"),
        ("c", gnomad_nmd_bar_chart, ".xlsx"),
        ("d", gnomad_rarity_and_constrain, ".csv"),
        ("e", gnomad_disease_genes, ".csv"),
        ("f", gnomad_disease_gene_enrichment, ".csv"),
        ("g", TCGA_TSG_OG, ".csv"),
        ("h", TCGA_cancer_genes, ".csv"),
    ],
    "supplementary": [
        #("NMDeff_distributions", NMDeff_distributions, ".xlsx"),
        #("preprocessing_stats", preprocessing_stats, ".xlsx"),
        #("hyperparameter_sweep", hyperparameter_sweep, ".csv"),
        #("long_exon_DMS_PTC_comparison_full", long_exon_DMS_PTC_comparison_full, ".xlsx"),
        #("top_long_exons", top_long_exons, ".xlsx"),
        #("cancer_gene_selection_scores", cancer_gene_selection_scores, ".csv"),
        #("penultimate_exon_genome_wide", penultimate_exon_genome_wide, ".xlsx"),
        #("SPreinit_ptc_curves", SPreinit_ptc_curves, ".csv"),
        #("SPreinit_ptc_curves_aug", SPreinit_ptc_curves_aug, ".csv"),
        #("context_downstream_trinuc", context_downstream_trinuc, ".csv"),
        ("gnomad_logodds_correlation", gnomad_logodds_correlation, ".csv"),
        ("DMS_interreplicate_correlations_full", DMS_interreplicate_correlations_full, ".xlsx"),
    ],
}


def _run_panel(
    figure_number: str,
    panel_letters: str,
    module,
    source_data_ext: str,
    regenerate: bool,
):
    """Run a single panel/script through the manuscript-app pipeline."""
    if figure_number == "supplementary":
        label = f"S_{panel_letters}"
    else:
        label = f"{figure_number}{panel_letters}"

    paths = get_paths(
        script_name=module.__name__.rsplit(".", 1)[-1],
        figure_label=label,
        figure_number=figure_number,
        source_data_ext=source_data_ext,
    )
    logger.info(f"Running {label} → {paths.figure_png}")
    module.main(
        figure_label=label,
        figure_number=figure_number,
        regenerate=regenerate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command(name="generate")
def generate_figs(
    fig: str = typer.Option(
        None,
        help="Generate a specific figure (e.g. Fig2, Fig7, supplementary). "
             "Omit to generate all.",
    ),
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        help="Force regeneration of source data even if it already exists.",
    ),
):
    """Generate manuscript figures (all or a specific figure)."""
    if fig is not None:
        if fig not in FIGURE_REGISTRY:
            logger.error(f"Unknown figure: {fig}. Available: {list(FIGURE_REGISTRY)}")
            raise typer.Exit(1)
        targets = {fig: FIGURE_REGISTRY[fig]}
    else:
        targets = FIGURE_REGISTRY

    for figure_number, panels in targets.items():
        logger.info(f"═══ Generating {figure_number} ═══")
        for panel_letters, module, ext in panels:
            try:
                _run_panel(figure_number, panel_letters, module, ext, regenerate)
                logger.success(f"  ✓ {figure_number}{panel_letters}")
            except Exception as e:
                logger.error(f"  ✗ {figure_number}{panel_letters}: {e}")
                raise


@app.command(name="panel")
def gen_panel(
    panel_id: str = typer.Argument(..., help="Panel ID (e.g., fig2a, Fig7c, fig2cde)"),
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate a specific panel."""
    import re
    
    # Parse panel_id to extract figure and panel letters
    # Expected format: fig2a, Fig7c, fig2cde, supplementarySomething, etc.
    match = re.match(r'(fig\d+|supplementary)([a-z0-9_]+)?', panel_id.lower())
    if not match:
        logger.error(f"Invalid panel ID format: {panel_id}. Expected format: fig2a, Fig7c, etc.")
        raise typer.Exit(1)
    
    figure_number = match.group(1)
    panel_letters = match.group(2) or ""
    
    # Convert fig2 -> Fig2 for consistency
    if figure_number.startswith("fig"):
        figure_number = "Fig" + figure_number[3:]
    
    if figure_number not in FIGURE_REGISTRY:
        logger.error(f"Unknown figure: {figure_number}. Available: {list(FIGURE_REGISTRY)}")
        raise typer.Exit(1)
    
    # Find the matching panel
    panels = FIGURE_REGISTRY[figure_number]
    for registered_letters, module, ext in panels:
        if registered_letters.lower() == panel_letters.lower():
            logger.info(f"═══ Generating {figure_number}{registered_letters} ═══")
            try:
                _run_panel(figure_number, registered_letters, module, ext, regenerate)
                logger.success(f"  ✓ {figure_number}{registered_letters}")
            except Exception as e:
                logger.error(f"  ✗ {figure_number}{registered_letters}: {e}")
                raise
            return
    
    # Panel not found
    available = [p[0] for p in panels]
    logger.error(f"Panel '{panel_letters}' not found in {figure_number}. Available panels: {available}")
    raise typer.Exit(1)


# ── Convenience per-figure commands ───────────────────────────────────────

@app.command(name="fig2")
def gen_fig2(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig2 (NMDetectiveAI) panels."""
    generate_figs(fig="Fig2", regenerate=regenerate)


@app.command(name="fig3")
def gen_fig3(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig3 (DMS overview) panels."""
    generate_figs(fig="Fig3", regenerate=regenerate)


@app.command(name="fig4")
def gen_fig4(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig4 (Penultimate exon) panels."""
    generate_figs(fig="Fig4", regenerate=regenerate)


@app.command(name="fig5")
def gen_fig5(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig5 (Long exon) panels."""
    generate_figs(fig="Fig5", regenerate=regenerate)


@app.command(name="fig6")
def gen_fig6(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig6 (Start-proximal reinitiation) panels."""
    generate_figs(fig="Fig6", regenerate=regenerate)


@app.command(name="fig7")
def gen_fig7(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig7 (Selection) panels."""
    generate_figs(fig="Fig7", regenerate=regenerate)


@app.command(name="fig8")
def gen_fig8(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all Fig8 (Local sequence context) panels."""
    generate_figs(fig="Fig8", regenerate=regenerate)


@app.command(name="supplementary")
def gen_supplementary(
    regenerate: bool = typer.Option(False, "--regenerate", help="Force source data regeneration."),
):
    """Generate all supplementary figures."""
    generate_figs(fig="supplementary", regenerate=regenerate)


@app.command(name="sp-supplementary-table")
def gen_sp_supplementary_table():
    """Combine start-proximal analysis tables into a single supplementary Excel file.

    Merges the following analysis tables (from TABLES_DIR/SP/) into an Excel
    workbook with one sheet per table:
      - loess_interpolated_matrix.csv
      - pca_matrix.csv
      - sigmoid_params_observations.csv
      - cluster_assignments.csv

    The output is saved to MANUSCRIPT_TABLES_DIR / start_proximal_analysis.xlsx.
    """
    from NMD.config import TABLES_DIR, MANUSCRIPT_TABLES_DIR

    sp_dir = TABLES_DIR / "SP"
    tables = {
        "LOESS_matrix": sp_dir / "loess_interpolated_matrix.csv",
        "PCA_matrix": sp_dir / "pca_matrix.csv",
        "Sigmoid_parameters": sp_dir / "sigmoid_params_observations.csv",
        "Cluster_assignments": sp_dir / "cluster_assignments.csv",
    }

    # Verify all tables exist
    missing = [name for name, path in tables.items() if not path.exists()]
    if missing:
        logger.error(
            f"Missing analysis tables: {missing}. Run the analysis scripts "
            "(dms_pca_analysis, dms_sigmoid_fitting, start_prox_clustering) first."
        )
        raise typer.Exit(1)

    output_path = MANUSCRIPT_TABLES_DIR / "start_proximal_analysis.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, csv_path in tables.items():
            df = pd.read_csv(csv_path, index_col=0 if sheet_name != "Cluster_assignments" else None)
            df.to_excel(writer, sheet_name=sheet_name)
            logger.info(f"  Added sheet '{sheet_name}' ({df.shape[0]} rows)")

    logger.success(f"Saved supplementary table to {output_path}")


if __name__ == "__main__":
    app()
