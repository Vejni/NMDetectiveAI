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
    'brca1_50nt': {
        'file': 'brca1_50ntss.csv',
        'title': 'Penultimate exon DMS experiment (BRCA1)',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (0, 0)
    },
    'long_exon': {
        'file': 'ONT_tbl.csv',
        'title': 'Long exon DMS experiment (BRCA1)',
        'fitness_cols': ['fitness_r1', 'fitness_r2', 'fitness_r3'],
        'position': (0, 1)
    },
    'genes_139': {
        'file': 'genes_139.csv',
        'title': 'Start proximal DMS experiment (122 genes)',
        'fitness_cols': ['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr'],
        'position': (0, 2)
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
        print(f"  After filtering high-error genes: {len(df)} rows")
        print(f"  Remaining genes: {df['gene'].unique()}")
    
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
    Create 1x3 grid of interreplicate correlation plots.
    """
    logger.info("Creating interreplicate correlation figure")
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Store all correlation results for summary table
    all_results = []
    
    # Store plot data for each panel
    panel_data = {}
    
    # Plot each dataset
    for dataset_name, config in DATASETS.items():
        row, col = config['position']
        ax = axes[col]

        # Load data
        df = load_dataset(dataset_name)

        # Calculate pairwise correlations
        correlations = calculate_pairwise_correlations(df, config['fitness_cols'])

        # Store data for this panel
        panel_rows = []

        # Overlay hexbin for all points in this dataset (all replicates pooled)
        # Pool all fitness columns for all pairwise combinations
        all_x = []
        all_y = []
        for corr in correlations:
            all_x.extend(corr['x'])
            all_y.extend(corr['y'])
        if len(all_x) > 0:
            hb = ax.hexbin(all_x, all_y, gridsize=60, cmap="Greys", bins='log', alpha=0.4, linewidths=0)

        # Plot each pairwise comparison
        for i, corr in enumerate(correlations):
            color = COLORS[i % len(COLORS)]
            alpha = 0.15 if len(correlations) > 1 else 0.3  # more transparent
            s = 8  # smaller point size

            # Determine replicate numbers (1-indexed)
            rep1_num = corr['rep1'].split('fitness')[-1].split('_')[0]  # Extract number from 'fitness1_uncorr'
            rep2_num = corr['rep2'].split('fitness')[-1].split('_')[0]  # Extract number from 'fitness2_uncorr'

            # Scatter plot (over hexbin)
            ax.scatter(corr['x'], corr['y'], 
                      c=color, alpha=alpha, s=s, marker='.',
                      label=f"Repl. {rep1_num} × Repl. {rep2_num}: ρ = {corr['spearman_rho']:.3f}, R² = {corr['r_squared']:.3f}")

            # Store results for summary table
            all_results.append({
                'dataset': dataset_name,
                'title': config['title'].replace('\n', ' '),
                'replicate_pair': f"{corr['rep1']} vs {corr['rep2']}",
                'n_points': corr['n_points'],
                'spearman_rho': corr['spearman_rho'],
                'r_squared': corr['r_squared'],
                'p_value': corr['p_value']
            })

            # Store x-y data for this replicate pair
            for x_val, y_val in zip(corr['x'], corr['y']):
                panel_rows.append({
                    'replicate_pair': f"Replicate {rep1_num} × Replicate {rep2_num}",
                    'replicate_1_RNA_abundance': x_val,
                    'replicate_2_RNA_abundance': y_val
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
        ax.set_xlabel('Replicate 1 RNA abundance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Replicate 2 RNA abundance', fontsize=14, fontweight='bold')
        ax.set_title(config['title'], fontsize=16, fontweight='bold')

        # Legend
        ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

        # Grid
        ax.grid(alpha=0.3, linestyle=':')

        # Tick label size
        ax.tick_params(axis='both', labelsize=12)

        # Store panel data
        panel_data[dataset_name] = pd.DataFrame(panel_rows)

        logger.info(f"  {dataset_name}: {len(correlations)} pairwise comparisons")
    
    plt.tight_layout()
    
    # Create summary table
    results_df = pd.DataFrame(all_results)
    
    return fig, results_df, panel_data


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
        source_data_ext=".xlsx",
    )
    logger.info("Starting DMS interreplicate correlation analysis")
    
    # Check if we can skip processing and load from existing source data
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        
        # Load data from Excel
        panel_data = {
            'brca1_50nt': pd.read_excel(paths.source_data, sheet_name='Panel_C_BRCA1_PE'),
            'long_exon': pd.read_excel(paths.source_data, sheet_name='Panel_D_LongExon'),
            'genes_139': pd.read_excel(paths.source_data, sheet_name='Panel_E_139genes_SP')
        }
        results_df = pd.read_excel(paths.source_data, sheet_name='Summary_Statistics')
        
        logger.info("Loaded source data successfully")
        
        # Regenerate only the figure (we need to recreate it from loaded data)
        # Create figure structure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # For each dataset, recreate the plot from loaded data
        for dataset_idx, (dataset_name, config) in enumerate(DATASETS.items()):
            row, col = config['position']
            ax = axes[col]
            
            # Get data for this panel
            data = panel_data[dataset_name]
            
            # Get unique replicate pairs
            replicate_pairs = data['replicate_pair'].unique()
            
            # Plot each replicate pair
            for i, rep_pair in enumerate(replicate_pairs):
                pair_data = data[data['replicate_pair'] == rep_pair]
                color = COLORS[i % len(COLORS)]
                alpha = 0.3 if len(replicate_pairs) > 1 else 0.6
                
                # Get statistics from results_df
                stats_row = results_df[
                    (results_df['dataset'] == dataset_name) & 
                    (results_df['replicate_pair'].str.contains(rep_pair.split('×')[0].strip().split()[-1]))
                ].iloc[i] if len(results_df) > 0 else None
                
                if stats_row is not None:
                    rho = stats_row['spearman_rho']
                    r2 = stats_row['r_squared']
                else:
                    rho = 0.0
                    r2 = 0.0
                
                ax.scatter(
                    pair_data['replicate_1_RNA_abundance'],
                    pair_data['replicate_2_RNA_abundance'],
                    c=color, alpha=alpha, s=20,
                    label=f"{rep_pair}: ρ = {rho:.3f}, R² = {r2:.3f}"
                )
            
            # Add diagonal line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1)
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            # Labels and title
            ax.set_xlabel('Replicate 1 RNA abundance', fontsize=14, fontweight='bold')
            ax.set_ylabel('Replicate 2 RNA abundance', fontsize=14, fontweight='bold')
            ax.set_title(config['title'], fontsize=16, fontweight='bold')
            
            # Legend
            ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
            
            # Grid
            ax.grid(alpha=0.3, linestyle=':')
            
            # Tick label size
            ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
    else:
        # Create and save figure + source data
        fig, results_df, panel_data = plot_interreplicate_grid()
        
        # Save source data to Excel with multiple sheets
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(paths.source_data, engine='openpyxl') as writer:
            # Panel C: BRCA1 Penultimate exon
            panel_data['brca1_50nt'].to_excel(writer, sheet_name='Panel_C_BRCA1_PE', index=False)
            
            # Panel D: Long exon
            panel_data['long_exon'].to_excel(writer, sheet_name='Panel_D_LongExon', index=False)
            
            # Panel E: 139 genes Start proximal
            panel_data['genes_139'].to_excel(writer, sheet_name='Panel_E_139genes_SP', index=False)
            
            # Summary statistics
            results_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Save figure
    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("DMS interreplicate correlation analysis complete!")


if __name__ == "__main__":
    main()
