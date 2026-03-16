"""
Interreplicate correlation for DMS experiments.

Creates a 2x3 grid showing correlation between biological replicates for:
- ATP7A penultimate exon (50nt rule)
- BRCA1 penultimate exon (50nt rule)
- ATP7A start proximal region
- BRCA1 start proximal region
- Multi-gene analysis (139 genes)
- Long exon analysis

For each subplot, shows all pairwise replicate comparisons with correlation statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths


# Configuration
SCRIPT_NAME = "DMS_interreplicate_correlations"
DMS_DIR = RAW_DATA_DIR / "DMS"

# Dataset configuration: (filename, title, fitness columns)
DATASETS = {
    'atp7a_50nt': {
        'file': 'ATP7A_50nts.csv',
        'title': 'ATP7A\nPenultimate exon',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr'],  # Only 2 replicates
        'position': (0, 0)
    },
    'brca1_50nt': {
        'file': 'brca1_50ntss.csv',
        'title': 'BRCA1\nPenultimate exon',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (1, 0)
    },
    'atp7a_spr': {
        'file': 'ATP7A_SPR.csv',
        'title': 'ATP7A\nStart proximal',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (0, 1)
    },
    'brca1_spr': {
        'file': 'BRCA1_SPR.csv',
        'title': 'BRCA1\nStart proximal',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (1, 1)
    },
    'genes_139': {
        'file': 'genes_139.csv',
        'title': '128 genes\nStart proximal',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (0, 2)
    },
    'long_exon': {
        'file': 'ONT_tbl.csv',
        'title': 'Long exon',
        'fitness_cols': ['fitness_r1', 'fitness_r2', 'fitness_r3'],
        'position': (1, 2)
    }
}

# Color palette for replicate comparisons
COLORS = ['#fb731d', '#4e79a7', '#59a14f']  # Orange, Blue, Green


def filter_high_error_genes(
    df: pd.DataFrame,
    sigma_col: str = "sigma",
    error_threshold: float = 0.5,
) -> pd.DataFrame:
    """Drop genes where more than *error_threshold* fraction of variants have sigma >= 1.

    Args:
        df: DataFrame containing ``gene`` and ``sigma_col`` columns.
        sigma_col: Column name for fitness error.
        error_threshold: Maximum allowed fraction of high-sigma variants per gene.

    Returns:
        Filtered DataFrame with high-error genes removed.
    """
    error_rate = df.groupby("gene")[sigma_col].apply(lambda s: (s >= 1).mean())
    high_error = error_rate[error_rate > error_threshold].index.tolist()
    logger.info(
        f"  Dropping {len(high_error)} high-error genes (>50% variants with sigma>=1): {high_error}"
    )
    logger.info(f"  Remaining genes: {df['gene'].nunique() - len(high_error)}")
    return df[~df["gene"].isin(high_error)].copy()


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load a DMS dataset.
    
    Args:
        dataset_name: Key in DATASETS dict
        
    Returns:
        DataFrame with fitness measurements
    """
    config = DATASETS[dataset_name]
    file_path = DMS_DIR / config['file']
    
    logger.info(f"Loading {dataset_name} from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df)} rows")
    
    # Check that fitness columns exist
    for col in config['fitness_cols']:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in {file_path}")
    
    # For multi-gene datasets, remove genes with high measurement error
    if dataset_name == 'genes_139':
        df = filter_high_error_genes(df)
    
    # Drop rows with missing fitness values
    df_clean = df.dropna(subset=config['fitness_cols'])
    logger.info(f"  After dropping NAs: {len(df_clean)} rows")
    
    return df_clean


def calculate_pairwise_correlations(df: pd.DataFrame, fitness_cols: list) -> list:
    """
    Calculate all pairwise Spearman correlations between replicates.
    
    Args:
        df: DataFrame with fitness columns
        fitness_cols: List of column names for fitness measurements
        
    Returns:
        List of dicts with correlation results
    """
    results = []
    n_reps = len(fitness_cols)
    
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            col1, col2 = fitness_cols[i], fitness_cols[j]
            
            # Get data for both replicates (drop any remaining NAs)
            mask = df[col1].notna() & df[col2].notna()
            x = df.loc[mask, col1].values
            y = df.loc[mask, col2].values
            
            # Calculate Spearman correlation
            rho, p_value = stats.spearmanr(x, y)
            
            # Calculate R²
            r2 = stats.pearsonr(x, y)[0] ** 2
            
            results.append({
                'rep1': col1,
                'rep2': col2,
                'n_points': len(x),
                'spearman_rho': rho,
                'p_value': p_value,
                'r_squared': r2,
                'x': x,
                'y': y
            })
    
    return results


def plot_interreplicate_grid():
    """
    Create 2x3 grid of interreplicate correlation plots.
    """
    logger.info("Creating interreplicate correlation figure")
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Store all correlation results for summary table
    all_results = []
    
    # Plot each dataset
    for dataset_name, config in DATASETS.items():
        row, col = config['position']
        ax = axes[row, col]
        
        # Load data
        df = load_dataset(dataset_name)
        
        # Calculate pairwise correlations
        correlations = calculate_pairwise_correlations(df, config['fitness_cols'])
        
        # Plot each pairwise comparison
        for i, corr in enumerate(correlations):
            color = COLORS[i % len(COLORS)]
            alpha = 0.5 if len(correlations) > 1 else 0.6
            
            # Scatter plot
            ax.scatter(corr['x'], corr['y'], 
                      c=color, alpha=alpha, s=20, 
                      label=f"ρ = {corr['spearman_rho']:.3f}")
            
            # Store results for table
            all_results.append({
                'dataset': dataset_name,
                'title': config['title'].replace('\n', ' '),
                'replicate_pair': f"{corr['rep1']} vs {corr['rep2']}",
                'n_points': corr['n_points'],
                'spearman_rho': corr['spearman_rho'],
                'r_squared': corr['r_squared'],
                'p_value': corr['p_value']
            })
        
        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Labels and title
        ax.set_xlabel('Replicate 1 fitness', fontsize=14)
        ax.set_ylabel('Replicate 2 fitness', fontsize=14)
        ax.set_title(config['title'], fontsize=16, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        
        # Grid
        ax.grid(alpha=0.3, linestyle=':')
        
        # Tick label size
        ax.tick_params(axis='both', labelsize=12)
        
        logger.info(f"  {dataset_name}: {len(correlations)} pairwise comparisons")
    
    plt.tight_layout()
    
    # Create summary table
    results_df = pd.DataFrame(all_results)
    
    return fig, results_df


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate DMS interreplicate correlation figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )
    logger.info("Starting DMS interreplicate correlation analysis")
    
    # Create and save figure + source data
    fig, results_df = plot_interreplicate_grid()
    
    # Save source data
    paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(paths.source_data, index=False)
    logger.info(f"Source data saved to {paths.source_data}")
    
    # Save figure
    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("DMS interreplicate correlation analysis complete!")


if __name__ == "__main__":
    main()
