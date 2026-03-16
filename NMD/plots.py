import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.metrics import confusion_matrix
import os

from NMD.config import TABLES_DIR, FIGURES_DIR, GENCODE_VERSION, VAL_CHRS, PROCESSED_DATA_DIR, MODELS_DIR
from NMD.utils import relative_squared_error, load_model
from loguru import logger
import typer
import torch

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from genome_kit import Genome

app = typer.Typer()


@app.command()
def analyse_predictions(file_path):
    # Load the dataframe
    df = pd.read_csv(file_path)

    # Extract the observed and predicted values
    observed = df["NMDeff"]
    predictions = df["predictions"]

    # Create a directory to save results (if it doesn't exist)
    file_name = (
        os.path.basename(file_path)
        .replace(".csv", "")
        .replace("_PTCs", "")
        .replace("_all", "")
        .replace("_confident_seq.", "_")
        .replace("_test_predictions", "")
    )
    output_dir = os.path.dirname(file_path)
    output_dir = os.path.join(output_dir, f"{file_name}")
    os.makedirs(output_dir, exist_ok=True)

    # 0. Move the original file to the output directory
    os.rename(file_path, output_dir + "/" + os.path.basename(file_path))

    # 1. Histogram of values with KDE
    plt.figure(figsize=(8, 6))
    sns.histplot(
        observed, kde=True, color="blue", label="Observed (NMDeff)", stat="density", bins=30
    )
    sns.histplot(
        predictions, kde=True, color="orange", label="Predictions", stat="density", bins=30
    )
    plt.axvline(x=0, color="red", linestyle="--", label="Zero")
    plt.title("Histogram and KDE of Observed and Predicted Values")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    hist_path = os.path.join(output_dir, "histogram_KDE.png")
    plt.savefig(hist_path)
    plt.close()

    # 2. Scatterplot of preds vs observed with line, correlation, and R2
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=observed, y=predictions)
    sns.lineplot(x=observed, y=observed, color="red", label="Perfect Fit")
    correlation = observed.corr(predictions, method='spearman')
    r2 = r2_score(observed, predictions)
    plt.title(
        f"Scatterplot of Predicted vs Observed\nSpearman R: {correlation:.3f}, R²: {r2:.3f}"
    )
    plt.xlabel("Observed (NMDeff)")
    plt.ylabel("Predictions")
    plt.legend()
    scatter_path = os.path.join(output_dir, "scatterplot.png")
    plt.savefig(scatter_path)
    plt.close()

    # 3. Confusion matrix
    observed_binary = (observed > 0).astype(int)
    predictions_binary = (predictions > 0).astype(int)
    cm = confusion_matrix(observed_binary, predictions_binary)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 4. Continuous ROC Curve, AUC
    fpr, tpr, _ = roc_curve(observed_binary, predictions)
    auc = roc_auc_score(observed_binary, predictions_binary)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve with Varying Thresholds")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # 5. Continuous PR Curve, AUC
    precision, recall, _ = precision_recall_curve(observed_binary, predictions)
    pr_auc = average_precision_score(observed_binary, predictions_binary)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.title("Precision-Recall Curve with Varying Thresholds")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path)
    plt.close()

    # 6. MSE and MAPE
    mse = mean_squared_error(observed, predictions)
    mape = mean_absolute_percentage_error(observed, predictions)
    rse = relative_squared_error(observed, predictions)

    # 7. Classification Metrics
    accuracy = accuracy_score(observed_binary, predictions_binary)
    f1 = f1_score(observed_binary, predictions_binary)
    mcc = matthews_corrcoef(observed_binary, predictions_binary)
    precision_bin = precision_score(observed_binary, predictions_binary)
    recall_bin = recall_score(observed_binary, predictions_binary)

    logger.info(
        f"\nContinuous Metrics: \nCorrelation: {correlation:.3f}, MSE: {mse:.3f}, MAPE: {mape:.3f}, RSE: {rse:.3f}, R²: {r2:.3f},"
        + f"\nBinary Metrics: \nAUC: {auc:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, Precision: {precision_bin:.3f}, Recall: {recall_bin:.3f}"
    )

    # Save performance metrics to a CSV
    metrics = {
        "Correlation": correlation,
        "MSE": mse,
        "MAPE": mape,
        "RSE": rse,
        "R²": r2,
        "ROC AUC": auc,
        "PR AUC": pr_auc,
        "Accuracy": accuracy,
        "F1": f1,
        "MCC": mcc,
        "Precision": precision_bin,
        "Recall": recall_bin,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(output_dir, "performance_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)


def plot_LE_predictions(test_df, output_path, col="NMDeff"):
    # Melt the dataframe to have fitness and predictions in a single column for easier plotting
    df_melted = test_df.melt(
        id_vars=["PTC", "stop_type", "sublib"],
        value_vars=[col, "predictions"],
        var_name="metric",
        value_name="value",
    )

    for codon in df_melted["stop_type"].unique():
        # Filter the melted dataframe for the current codon
        df_codons = df_melted[df_melted["stop_type"] == codon]

        # order by sublib then PTC
        df_codons = df_codons.sort_values(by=["sublib", "PTC"])
        # Ensure the 'sublib' column is treated as a categorical variable
        df_codons["sublib"] = pd.Categorical(df_codons["sublib"], ordered=True)

        # Create the plot
        g = sns.relplot(
            data=df_codons,
            x="PTC",
            y="value",
            style="metric",
            kind="scatter",  # Changed from 'line' to 'scatter'
            col="sublib", 
            col_wrap=3,  # Adjust the number of columns in the facet grid
            facet_kws={"sharey": False, "sharex": False},  # Ensure x-axis ticks are not fixed
            height=4,
            aspect=1.5,
        )

        # save the plot
        g.set_axis_labels("PTC Position", "Fitness / Predictions")
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle("LE Predictions vs Fitness by PTC Position and Stop Type", y=1.02)
        g.savefig(str(output_path).replace(".png", f"_{codon}.png"), bbox_inches="tight")


def plot_predictions(test_df, output_path):
    """Plot scatter of predicted vs actual values"""
    plt.figure(figsize=(8, 8))

    observed = test_df["NMDeff"]
    predicted = test_df["predictions"]

    # Create scatter plot colored by gene
    if "gene" in test_df.columns:
        for gene in test_df["gene"].unique():
            gene_data = test_df[test_df["gene"] == gene]
            plt.scatter(gene_data["NMDeff"], gene_data["predictions"], alpha=0.6)
    elif "set" in test_df.columns:
        for set_name in test_df["set"].unique():
            set_data = test_df[test_df["set"] == set_name]
            plt.scatter(set_data["NMDeff"], set_data["predictions"], alpha=0.6, label=set_name)
        plt.legend(title="Set")
    elif "chr" in test_df.columns:
        for chr_type, color in [("VAL", "#3d405b"), ("TEST", "#e07a5f"), ("TRAIN", "#81b29a")]:
            if chr_type == "VAL":
                chr_data = test_df[test_df["chr"].isin(VAL_CHRS)]
            else:  # TRAIN
                chr_data = test_df[~test_df["chr"].isin(VAL_CHRS)]
            
            if len(chr_data) > 0:
                plt.scatter(chr_data["NMDeff"], chr_data["predictions"], 
                           alpha=0.6, color=color, label=f"{chr_type} chromosomes")
        plt.legend(title="Chromosome Set")
    else:
        plt.scatter(observed, predicted, alpha=0.6)

    # Add diagonal line
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

    # Calculate metrics
    r2 = r2_score(observed, predicted)
    corr = observed.corr(predicted, method='spearman')

    plt.xlabel("Observed NMD Efficiency")
    plt.ylabel("Predicted NMD Efficiency")
    plt.title(f"Predicted vs Observed\nR² = {r2:.3f}, Spearman R = {corr:.3f}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved predictions plot to {output_path}")
    plt.close()

    return r2, corr


def plot_gene_ptc_fitness(
    test_df, gene, output_path, fold=None, frac=0.3, col="fitness_gene_specific"
):
    """Plot and save fitness vs PTC position for a single gene"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get data for this gene
    gene_data = test_df[test_df["gene"] == gene].sort_values("PTCposition")

    # Calculate correlations
    r2 = r2_score(gene_data[col], gene_data["predictions"])
    pearson = np.corrcoef(gene_data[col], gene_data["predictions"])[0, 1]
    spearman = gene_data[col].corr(gene_data["predictions"], method="spearman")
    rmse = np.sqrt(np.mean((gene_data[col] - gene_data["predictions"]) ** 2))
    mape = mean_absolute_percentage_error(gene_data[col], gene_data["predictions"])
    rse = relative_squared_error(gene_data[col], gene_data["predictions"])

    # Create scatter plot
    ax.scatter(
        gene_data["PTCposition"],
        gene_data["predictions"],
        color="blue",
        label="Predicted",
        alpha=0.3,
    )
    ax.errorbar(
        gene_data["PTCposition"],
        gene_data[col],
        yerr=gene_data["sigma"],
        color="red",
        label="Observed",
        fmt="o",
        alpha=0.3,
        capsize=3,
        elinewidth=1,
        capthick=1,
    )

    # Calculate and plot LOESS curves
    pred_smooth = lowess(gene_data["predictions"], gene_data["PTCposition"], frac=frac)
    obs_smooth = lowess(gene_data[col], gene_data["PTCposition"], frac=frac)

    ax.plot(
        pred_smooth[:, 0], pred_smooth[:, 1], color="blue", linewidth=2, label="Predicted (LOESS)"
    )
    ax.plot(obs_smooth[:, 0], obs_smooth[:, 1], color="red", linewidth=2, label="Observed (LOESS)")

    # Add correlation metrics as text
    ax.text(
        0.02,
        0.98,
        f"R² = {r2:.3f}\nPearson = {pearson:.3f}\nSpearman = {spearman:.3f}\nRMSE = {rmse:.3f}\nMAPE = {mape:.3f}\nRSE = {rse:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Customize the plot
    ax.set_xlabel("PTC Position")
    ax.set_ylabel(col)
    fold_text = f" (Fold {fold})" if fold is not None else ""
    ax.set_title(f"{gene}{fold_text}\nn = {len(gene_data)}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return r2, pearson, spearman, rmse, mape, rse


def plot_gene_metrics_distribution(metrics_df, output_path):
    """Plot violin plots of gene metrics across all folds"""
    # Check which metrics are available in the DataFrame
    available_metrics = []
    metric_mappings = {
        "r_squared": "R² Score",
        "correlation": "Correlation", 
        "mse": "MSE"
    }
    
    for metric, title in metric_mappings.items():
        if metric in metrics_df.columns:
            available_metrics.append((metric, title))
    
    if not available_metrics:
        logger.warning("No recognized metrics found in DataFrame for violin plot")
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 5))
    
    # Handle case where there's only one metric
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, available_metrics):
        data = metrics_df[metric].dropna()
        if len(data) > 0:
            ax.violinplot(data)
            ax.set_title(
                f"{title}\nMean: {data.mean():.3f}\nMedian: {data.median():.3f}"
            )
        else:
            ax.set_title(f"{title}\nNo data")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved gene metrics distribution plot to {output_path}")


def plot_dms_sp_correlation_distribution(all_gene_metrics, output_path):
    """
    Plot violin plot of DMS_SP gene-level correlation distributions across models.
    
    Args:
        all_gene_metrics (dict): Dictionary with model names as keys and gene metrics DataFrames as values
        output_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for violin plot
    correlation_data = []
    model_names = []
    
    for model_name, gene_metrics in all_gene_metrics.items():
        if 'correlation' in gene_metrics.columns:
            # Remove NaN values
            corr_values = gene_metrics['correlation'].dropna()
            correlation_data.append(corr_values.values)
            model_names.append(model_name)
            logger.info(f"{model_name}: {len(corr_values)} genes, "
                       f"median correlation = {corr_values.median():.3f}")
    
    if not correlation_data:
        logger.warning("No correlation data found for violin plot")
        return
    
    # Create violin plot
    violin_parts = ax.violinplot(correlation_data, positions=range(1, len(correlation_data) + 1))
    
    # Customize violin plot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(correlation_data)))
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add median points
    medians = [np.median(data) for data in correlation_data]
    ax.scatter(range(1, len(correlation_data) + 1), medians, 
               color='red', s=50, zorder=3, label='Median')
    
    # Customize plot
    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Gene-level Correlation')
    ax.set_title('DMS_SP Gene-level Correlation Distribution Across Models')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics as text
    stats_text = ""
    for i, (model_name, data) in enumerate(zip(model_names, correlation_data)):
        stats_text += f"{model_name}:\n  n={len(data)}, median={np.median(data):.3f}\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved DMS_SP correlation distribution plot to {output_path}")


def plot_dms_sp_position_averaged(predictions_df, output_path, label_col="NMDeff_Norm"):
    """
    Plot mean observations and predictions across genes at each PTC position for DMS_SP,
    with confidence intervals.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with columns 'PTCposition', label_col, 'predictions', 'gene'
        output_path (str): Path to save the plot
        label_col (str): Column name for observed values
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check required columns
    required_cols = ['PTCposition', label_col, 'predictions']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for position-averaged plot: {missing_cols}")
        return
    
    # Group by PTC position and calculate statistics
    position_stats = predictions_df.groupby('PTCposition').agg({
        label_col: ['mean', 'sem', 'count'],
        'predictions': ['mean', 'sem', 'count']
    }).round(4)
    
    # Flatten column names
    position_stats.columns = ['_'.join(col).strip() for col in position_stats.columns]
    position_stats = position_stats.reset_index()
    
    # Calculate 95% confidence intervals
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    # For observed values
    position_stats['obs_ci_lower'] = position_stats[f'{label_col}_mean'] - \
        stats.t.ppf(1 - alpha/2, position_stats[f'{label_col}_count'] - 1) * position_stats[f'{label_col}_sem']
    position_stats['obs_ci_upper'] = position_stats[f'{label_col}_mean'] + \
        stats.t.ppf(1 - alpha/2, position_stats[f'{label_col}_count'] - 1) * position_stats[f'{label_col}_sem']
    
    # For predictions
    position_stats['pred_ci_lower'] = position_stats['predictions_mean'] - \
        stats.t.ppf(1 - alpha/2, position_stats['predictions_count'] - 1) * position_stats['predictions_sem']
    position_stats['pred_ci_upper'] = position_stats['predictions_mean'] + \
        stats.t.ppf(1 - alpha/2, position_stats['predictions_count'] - 1) * position_stats['predictions_sem']
    
    # Sort by position for proper line plotting
    position_stats = position_stats.sort_values('PTCposition')
    
    # Plot observed values with CI
    ax.plot(position_stats['PTCposition'], position_stats[f'{label_col}_mean'], 
            'o-', color='blue', linewidth=2, markersize=4, label='Observed (mean)', alpha=0.8)
    ax.fill_between(position_stats['PTCposition'], 
                    position_stats['obs_ci_lower'], 
                    position_stats['obs_ci_upper'],
                    alpha=0.3, color='blue', label='Observed 95% CI')
    
    # Plot predictions with CI
    ax.plot(position_stats['PTCposition'], position_stats['predictions_mean'], 
            's-', color='red', linewidth=2, markersize=4, label='Predicted (mean)', alpha=0.8)
    ax.fill_between(position_stats['PTCposition'], 
                    position_stats['pred_ci_lower'], 
                    position_stats['pred_ci_upper'],
                    alpha=0.3, color='red', label='Predicted 95% CI')
    
    # Calculate overall correlation between position-averaged values
    overall_corr = np.corrcoef(position_stats[f'{label_col}_mean'], 
                              position_stats['predictions_mean'])[0, 1]
    
    # Calculate R² for position-averaged values
    from sklearn.metrics import r2_score
    overall_r2 = r2_score(position_stats[f'{label_col}_mean'], 
                         position_stats['predictions_mean'])
    
    # Customize plot
    ax.set_xlabel('PTC Position (Codon)', fontsize=12)
    ax.set_ylabel('NMD Efficiency (Position-averaged)', fontsize=12)
    ax.set_title(f'DMS_SP: Position-averaged Observations vs Predictions\n'
                f'Correlation: {overall_corr:.3f}, R²: {overall_r2:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add statistics as text
    n_positions = len(position_stats)
    total_observations = predictions_df.shape[0]
    n_genes = predictions_df['gene'].nunique() if 'gene' in predictions_df.columns else 'Unknown'
    
    stats_text = f'Positions: {n_positions}\n'
    stats_text += f'Total observations: {total_observations}\n'
    stats_text += f'Genes: {n_genes}\n'
    stats_text += f'Avg obs/position: {total_observations/n_positions:.1f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved DMS_SP position-averaged plot to {output_path}")
    logger.info(f"Position-averaged correlation: {overall_corr:.3f}, R²: {overall_r2:.3f}")
    
    return position_stats


def plot_PE_predictions(preds, output_path, col="NMDeff"):
    # Create faceted scatter plot of PTC_pos_rev vs predictions and observations
    fig, axes = plt.subplots(1, len(preds["gene"].unique()), figsize=(15, 5), sharey=True)

    genes = preds["gene"].unique()

    for i, gene in enumerate(genes):
        ax = axes[i] if len(genes) > 1 else axes

        # Filter data for current gene
        gene_data = preds[preds["gene"] == gene]

        # Plot predictions and observations
        ax.scatter(gene_data["PTC_pos_rev"], gene_data["predictions"], 
                  label="Predicted", alpha=0.7, marker='o')
        ax.scatter(gene_data["PTC_pos_rev"], gene_data[col], 
                  label="Observed", alpha=0.7, marker='s')

        ax.set_xlabel("PTC Position (Reverse)")
        ax.set_ylabel("NMD Efficiency")
        ax.set_title(f"{gene}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


@app.command()
def plot_gene_cv_results(
    input_file: str = TABLES_DIR / "gene_cv_results.csv",
    output_path: str = FIGURES_DIR / "gene_cv_results.png",
):
    """Plot the cross-validation results for each gene with test metrics by Tags"""
    cv_results = pd.read_csv(input_file)

    # Define the 5 test metrics
    metrics = ["r2", "corr", "mse", "rse"]

    # Create subplots for each metric
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    for i, metric in enumerate(metrics):
        sns.boxplot(data=cv_results, x="Tags", y=f"test_genes_median_{metric}", ax=axes[i])
        axes[i].set_title(f"{metric.upper()} by Tags")
        axes[i].set_xlabel("Tags")
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_transcript_ptc_predictions(transcript_id, gene_name, ptc_positions, predictions, 
                                    output_path=None, show_exon_boundaries=True, 
                                    observed_datasets=None, show_observations=True, gencode_version=GENCODE_VERSION):
    """
    Plot PTC predictions across a transcript with exon boundary annotations and optional observed data.
    
    Args:
        transcript_id (str): Transcript ID
        gene_name (str): Gene name
        ptc_positions (list): List of PTC positions (codon numbers)
        predictions (list): List of predicted NMD efficiencies
        output_path (str, optional): Path to save the plot
        show_exon_boundaries (bool): Whether to show exon boundaries
        observed_datasets (dict, optional): Dictionary with dataset DataFrames. 
                                          If None, will load default TCGA/GTEx datasets
        show_observations (bool): Whether to show observed data points
        gencode_version (str): GENCODE version to use for genome initialization
    Returns:
        tuple: (figure, axis)
    """
    from genome_kit import Genome
    import pandas as pd
    from NMD.config import PROCESSED_DATA_DIR
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot model predictions
    ax.plot(ptc_positions, predictions, 'b-o', markersize=4, linewidth=2, 
            label='Model Predictions', alpha=0.8)
    
    # Load and plot observed data if requested
    if show_observations:
        # Load datasets if not provided
        if observed_datasets is None:
            somatic_TCGA = pd.read_csv(PROCESSED_DATA_DIR / "PTC/somatic_TCGA.stopgain.csv")
            germline_TCGA = pd.read_csv(PROCESSED_DATA_DIR / "PTC/germline_TCGA.stopgain.csv")
            GTEx = pd.read_csv(PROCESSED_DATA_DIR / "PTC/GTEx.stopgain.csv")
            
            somatic_TCGA["dataset"] = "Somatic TCGA"
            germline_TCGA["dataset"] = "Germline TCGA"
            GTEx["dataset"] = "GTEx"
            
            observed_datasets = {
                "somatic_TCGA": somatic_TCGA,
                "germline_TCGA": germline_TCGA,
                "GTEx": GTEx
            }
        
        if show_observations:
            # Extract base transcript ID (remove version number)
            base_transcript_id = transcript_id.split(".")[0]
            
            # Colors for different datasets
            dataset_colors = {
                "Somatic TCGA": "red",
                "Germline TCGA": "orange", 
                "GTEx": "green"
            }
            
            # Plot observations from each dataset
            for dataset_name, dataset_df in observed_datasets.items():
                if "dataset" in dataset_df.columns:
                    dataset_label = dataset_df["dataset"].iloc[0]
                else:
                    dataset_label = dataset_name
                
                # Filter for matching transcript
                transcript_data = dataset_df[dataset_df["transcript_id"] == base_transcript_id]
                
                if len(transcript_data) > 0:
                    # Plot observed NMD efficiency vs variant position
                    ax.scatter(transcript_data["varpos"], transcript_data["ASE_NMD_efficiency_TPM_norm"],
                              color=dataset_colors.get(dataset_label, "purple"), 
                              alpha=0.7, s=50, label=f'{dataset_label} Observed',
                              marker='s', edgecolor='black', linewidth=0.5)
                    
                    logger.info(f"Found {len(transcript_data)} observations in {dataset_label} for {base_transcript_id}")
    
    # Add exon boundaries if requested
    if show_exon_boundaries:
        # Get transcript information to calculate exon boundaries
        genome = Genome(gencode_version)
        transcript = genome.transcripts[transcript_id]
        
        if transcript is not None and hasattr(transcript, 'exons') and hasattr(transcript, 'cdss'):
            # Calculate exon boundaries in transcript nucleotide coordinates
            utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
            
            # Get exon boundaries within CDS
            cumulative_length = 0
            exon_boundaries = []
            
            for exon in transcript.exons:
                cumulative_length += len(exon)
                # Convert to transcript coordinates (matching varpos calculation)
                if cumulative_length > utr5_length:
                    # This boundary is within or after CDS
                    transcript_boundary = cumulative_length
                    if transcript_boundary <= max(ptc_positions):
                        exon_boundaries.append(transcript_boundary)
            
            # Plot exon boundaries
            for i, boundary in enumerate(exon_boundaries[:-1]):  # Exclude last boundary
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                if i == 0:
                    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1, 
                                label='Exon boundaries')
            
            # Highlight the penultimate exon junction if it exists
            if len(exon_boundaries) >= 2:
                penultimate_junction = exon_boundaries[-2]
                if penultimate_junction <= max(ptc_positions):
                    ax.axvline(x=penultimate_junction, color='red', linestyle='-', alpha=0.8, 
                                linewidth=2, label='Last exon junction')
    
    # Customize the plot
    ax.set_xlabel('PTC Position (Nucleotide in Transcript)', fontsize=12)
    ax.set_ylabel('Predicted NMD Efficiency', fontsize=12)
    ax.set_title(f'PTC Predictions for {gene_name} (Transcript: {transcript_id})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add some statistics as text
    stats_text = f'Total PTCs: {len(ptc_positions)}\n'
    stats_text += f'Mean Pred: {np.mean(predictions):.3f}\n'
    stats_text += f'Std Pred: {np.std(predictions):.3f}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved transcript PTC plot to {output_path}")
    
    return fig, ax


def plot_model_comparison_gene_level(all_gene_metrics, output_path=None):
    """Plot comparison of gene-level metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(['correlation', 'r_squared', 'mse']):
        ax = axes[i]
        data_to_plot = [
            all_gene_metrics[model][metric].dropna()
            for model in all_gene_metrics.keys()
        ]
        
        ax.boxplot(data_to_plot, labels=list(all_gene_metrics.keys()))
        ax.set_title(f'Gene-level {metric.replace("_", " ").title()} Distribution')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved gene-level comparison plot to {output_path}")
    
    return fig


def plot_model_comparison_subsets(all_subset_results, output_path=None):
    """Plot comparison of subset-level metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    subsets = all_subset_results[list(all_subset_results.keys())[0]]['subset'].values
    n_models = len(all_subset_results)
    x = np.arange(len(subsets))
    width = 0.8 / n_models
    
    for i, metric in enumerate(['correlation', 'r_squared', 'mse']):
        ax = axes[i]
        
        for j, (model_name, results_df) in enumerate(all_subset_results.items()):
            values = results_df[metric].values
            ax.bar(x + j * width, values, width, label=model_name)
        
        ax.set_title(f'Subset-level {metric.replace("_", " ").title()} Comparison')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel('Subset')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(subsets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric == 'mse':
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subset comparison plot to {output_path}")
    
    return fig


def plot_single_model_results(predictions_df, label_col, model_name, output_path=None, include_test=True):
    """Create scatter plot for a single model's predictions with performance metrics by dataset split."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    true_values = predictions_df[label_col]
    predictions = predictions_df['predictions']
    
    # Calculate overall metrics
    overall_corr = np.corrcoef(true_values, predictions)[0, 1]
    overall_r2 = r2_score(true_values, predictions)
    
    # Color by chromosome sets and calculate split-specific metrics
    metrics_text = f"ALL: Corr={overall_corr:.3f}, R²={overall_r2:.3f}, n={len(predictions_df)}\n"
    
    if 'chr' in predictions_df.columns:
        if include_test:
            chr_splits = [("TRAIN", "#81b29a"), ("VAL", "#3d405b"), ("TEST", "#e07a5f")]
        else:
            chr_splits = [("TRAIN", "#81b29a"), ("VAL", "#3d405b")]
        
        for chr_type, color in chr_splits:
            if chr_type == "VAL":
                chr_data = predictions_df[predictions_df["chr"].isin(VAL_CHRS)]
            else:  # TRAIN
                if include_test:
                    chr_data = predictions_df[~predictions_df["chr"].isin(VAL_CHRS)]
                else:
                    # When test is disabled, train includes test chromosomes
                    chr_data = predictions_df[~predictions_df["chr"].isin(VAL_CHRS)]
            
            if len(chr_data) > 0:
                ax.scatter(chr_data[label_col], chr_data['predictions'], 
                          alpha=0.6, color=color, label=f"{chr_type} chromosomes", s=30)
                
                # Calculate metrics for this split
                split_corr = chr_data[label_col].corr(chr_data['predictions'], method='spearman')
                split_r2 = r2_score(chr_data[label_col], chr_data['predictions'])
                metrics_text += f"{chr_type}: Spearman R={split_corr:.3f}, R²={split_r2:.3f}, n={len(chr_data)}\n"
        
        ax.legend(title="Chromosome Set")
    else:
        ax.scatter(true_values, predictions, alpha=0.6, s=30)
    
    # Add perfect fit line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Fit')
    
    # Add metrics text box
    ax.text(0.02, 0.98, metrics_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_xlabel(f'True Values ({label_col})')
    ax.set_ylabel('Predictions')
    ax.set_title(f'{model_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model plot to {output_path}")
    
    return fig


def plot_nmd_efficiency_distributions(df, output_path):
    """
    Plot NMD analysis with boxplot and histogram for different variant types.
    
    Args:
        df: DataFrame containing NMD data
    """
    col = "NMDeff"

    # Create a copy and remap variant types
    df_plot = df.copy()
    df_plot['stopgain'] = df_plot['stopgain'].replace({
        'nonsense': 'SNV',
        'frameshift_insertion': 'FS',
        'frameshift_deletion': 'FS'
    })
    
    # Check if we have FS variants
    has_fs = (df_plot['stopgain'] == 'FS').any()
    
    # Define the variant types based on what's present
    variant_types = ['SNV']
    if has_fs:
        variant_types.append('FS')
    
    # Determine number of rows based on variant types present
    n_rows = len(variant_types)
    
    # Calculate global x-axis limits for first column
    col_data = df_plot[col].dropna()
    x_min, x_max = col_data.min(), col_data.max()
    x_margin = (x_max - x_min) * 0.05  # 5% margin
    x_limits = (x_min - x_margin, x_max + x_margin)
    
    # Calculate global y-axis limits for second column (boxplots)
    all_boxplot_data = []
    for variant_type in variant_types:
        variant_df = df_plot[df_plot['stopgain'] == variant_type]
        all_boxplot_data.extend(variant_df[col].dropna().values)
    
    if all_boxplot_data:
        y_min, y_max = min(all_boxplot_data), max(all_boxplot_data)
        y_margin = (y_max - y_min) * 0.05
        y_limits = (y_min - y_margin, y_max + y_margin)
    else:
        y_limits = None
    
    # Create figure with appropriate number of rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))
    
    # Ensure axes is 2D even with single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # First column: Overall NMDeff distribution with triggering/evading groups
    for i, variant_type in enumerate(variant_types):
        ax_overall = axes[i, 0]
        
        # Get all data for this variant type
        variant_df = df_plot[df_plot['stopgain'] == variant_type]
        
        # Define triggering and evading groups
        evading_mask = variant_df["Last_Exon"] == 1
        triggering_mask = (variant_df["Last_Exon"] == 0) & (variant_df["Penultimate_Exon"] == 0) & \
                         (variant_df["Start_Prox"] == 0) & (variant_df["Long_Exon"] == 0)
        
        evading_group = variant_df[evading_mask]
        triggering_group = variant_df[triggering_mask]
        
        # Colors matching the KDE plot style
        color_triggering = '#022778'  # Dark blue for NMD-triggering
        color_evading = '#ff9e9d'     # Light pink for NMD-evading
        
        # Plot histogram + KDE for all data
        if col in variant_df.columns:
            # Histogram for all data
            ax_overall.hist(variant_df[col].dropna(), bins=30, alpha=0.3, 
                           color='gray', density=True, label='All variants')
            
            # KDE for triggering group
            if len(triggering_group) > 0:
                sns.kdeplot(data=triggering_group, x=col, ax=ax_overall,
                           label='NMD-triggering', color=color_triggering, 
                           bw_adjust=0.5, linewidth=3)
                triggering_mean = triggering_group[col].mean()
                ax_overall.axvline(x=triggering_mean, color=color_triggering, 
                                 linestyle='--', alpha=0.8, linewidth=2)
            
            # KDE for evading group
            if len(evading_group) > 0:
                sns.kdeplot(data=evading_group, x=col, ax=ax_overall,
                           label='NMD-evading', color=color_evading, 
                           bw_adjust=0.5, linewidth=3)
                evading_mean = evading_group[col].mean()
                ax_overall.axvline(x=evading_mean, color=color_evading, 
                                 linestyle='--', alpha=0.8, linewidth=2)
            
            ax_overall.set_title(f'{variant_type} - {col} Distribution', 
                               fontsize=14, fontweight='bold')
            ax_overall.set_xlabel(col, fontsize=12)
            ax_overall.set_ylabel('Density', fontsize=12)
            ax_overall.legend(fontsize=10)
            ax_overall.grid(True, alpha=0.3)
            ax_overall.set_xticks(range(int(x_limits[0]), int(x_limits[1])+1))
            ax_overall.set_xlim(x_limits)  # Fixed x-axis scale
    
    # Second column: Boxplot
    for i, variant_type in enumerate(variant_types):
        variant_df = df_plot[df_plot['stopgain'] == variant_type]
        
        # Define groups for this variant type
        last_exon = variant_df[variant_df["Last_Exon"] == 1]
        penultimate_exon = variant_df[(variant_df["Last_Exon"] == 0) & (variant_df["Penultimate_Exon"] == 1)]
        start_prox = variant_df[(variant_df["Last_Exon"] == 0) & (variant_df["Penultimate_Exon"] == 0) & (variant_df["Start_Prox"] == 1)]
        long_exon = variant_df[(variant_df["Last_Exon"] == 0) & (variant_df["Penultimate_Exon"] == 0) & (variant_df["Start_Prox"] == 0) & (variant_df["Long_Exon"] == 1)]
        nmd_triggering = variant_df[(variant_df["Last_Exon"] == 0) & (variant_df["Penultimate_Exon"] == 0) & (variant_df["Start_Prox"] == 0) & (variant_df["Long_Exon"] == 0)]
        
        # Boxplot
        ax1 = axes[i, 1]
        groups_data = [
            last_exon[col].dropna(),
            penultimate_exon[col].dropna(),
            start_prox[col].dropna(),
            long_exon[col].dropna(),
            nmd_triggering[col].dropna()
        ]
        
        group_labels = ['Last Exon', 'Penultimate Exon', 'Start Proximal', 'Long Exon', 'NMD Triggering']
        colors = ['#ff9e9d', '#ffb3b3', '#ffc8c8', '#ffdddd', '#022778']
        
        bp = ax1.boxplot(groups_data, labels=group_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title(f'{variant_type} - Distribution of {col} by NMD Groups', fontsize=14, fontweight='bold')
        ax1.set_ylabel(col, fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Set fixed y-axis scale for boxplots
        if y_limits:
            ax1.set_ylim(y_limits)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dms_sp_normalization_comparison(ptc_data, dms_data, output_path, df_col="NMDeff"):
    """
    Plot comparison of PTC dataset vs normalized DMS dataset with LOESS fits and distributions.
    Shows before and after normalization in a 2x2 grid (same format as PE dataset).
    
    Args:
        ptc_data (pd.DataFrame): PTC dataset with df_col and PTC_CDS_pos
        dms_data (pd.DataFrame): DMS dataset with NMDeff, NMDeff_Norm, and PTCposition
        output_path (str): Path to save the plot
        dataset_name (str): Name of the DMS dataset for labeling
    """
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Distribution comparison (before and after)
    ax_dist_before = fig.add_subplot(gs[0, 0])
    ax_dist_after = fig.add_subplot(gs[0, 1])
    
    # Bottom row: LOESS fits (before and after)
    ax_loess_before = fig.add_subplot(gs[1, 0])
    ax_loess_after = fig.add_subplot(gs[1, 1])
    
    # === Top Left: Distribution Before Normalization ===
    ax_dist_before.hist(dms_data['NMDeff'].dropna(), bins=50, alpha=0.6, density=True,
                        color='orange', label='DMS (original)')
    ax_dist_before.hist(ptc_data[df_col].dropna(), bins=50, alpha=0.6, density=True,
                        color='blue', label='PTC')
    sns.kdeplot(data=dms_data, x='NMDeff', color='darkorange',
                linewidth=2, ax=ax_dist_before, label='DMS KDE')
    sns.kdeplot(data=ptc_data, x=df_col, color='darkblue',
                linewidth=2, ax=ax_dist_before, label='PTC KDE')
    ax_dist_before.set_xlabel('NMDeff')
    ax_dist_before.set_ylabel('Density')
    ax_dist_before.set_title('Distribution Before Normalization')
    ax_dist_before.legend()
    ax_dist_before.grid(True, alpha=0.3)
    
    # === Top Right: Distribution After Normalization ===
    ax_dist_after.hist(dms_data['NMDeff_Norm'].dropna(), bins=50, alpha=0.6, density=True,
                       color='green', label='DMS (normalized)')
    ax_dist_after.hist(ptc_data[df_col].dropna(), bins=50, alpha=0.6, density=True,
                       color='blue', label='PTC')
    dms_temp = dms_data.copy()
    dms_temp['NMDeff'] = dms_temp['NMDeff_Norm']
    sns.kdeplot(data=dms_temp, x='NMDeff', color='darkgreen',
                linewidth=2, ax=ax_dist_after, label='DMS KDE')
    sns.kdeplot(data=ptc_data, x=df_col, color='darkblue',
                linewidth=2, ax=ax_dist_after, label='PTC KDE')
    ax_dist_after.set_xlabel('NMDeff')
    ax_dist_after.set_ylabel('Density')
    ax_dist_after.set_title('Distribution After Position-Dependent Normalization')
    ax_dist_after.legend()
    ax_dist_after.grid(True, alpha=0.3)
    
    # === Bottom Left: LOESS Fits Before Normalization ===
    # Fit LOESS to PTC data
    ptc_clean = ptc_data.dropna(subset=['PTC_CDS_pos', df_col])
    ptc_smooth = lowess(ptc_clean[df_col], ptc_clean['PTC_CDS_pos'], frac=0.4)
    
    # Fit LOESS to DMS original data
    dms_clean = dms_data.dropna(subset=['PTCposition_nt', 'NMDeff'])
    dms_smooth = lowess(dms_clean['NMDeff'], dms_clean['PTCposition_nt'], frac=0.4)
    
    ax_loess_before.scatter(ptc_clean['PTC_CDS_pos'], ptc_clean[df_col],
                            alpha=0.3, s=15, color='blue', label='PTC TCGA')
    ax_loess_before.plot(ptc_smooth[:, 0], ptc_smooth[:, 1],
                         color='darkblue', linewidth=3, label='PTC LOESS fit')
    ax_loess_before.scatter(dms_clean['PTCposition_nt'], dms_clean['NMDeff'],
                            alpha=0.3, s=15, color='orange', label='DMS (original)')
    ax_loess_before.plot(dms_smooth[:, 0], dms_smooth[:, 1],
                         color='darkorange', linewidth=3, linestyle='--',
                         label='DMS LOESS fit (original)')
    ax_loess_before.set_xlabel('PTC Position (nt)')
    ax_loess_before.set_ylabel('NMDeff')
    ax_loess_before.set_title('LOESS Fits Before Normalization')
    ax_loess_before.legend()
    ax_loess_before.grid(True, alpha=0.3)
    
    # === Bottom Right: LOESS Fits After Normalization ===
    # Fit LOESS to normalized DMS data
    dms_norm_clean = dms_data.dropna(subset=['PTCposition_nt', 'NMDeff_Norm'])
    dms_norm_smooth = lowess(dms_norm_clean['NMDeff_Norm'], dms_norm_clean['PTCposition_nt'], frac=0.4)
    
    ax_loess_after.scatter(ptc_clean['PTC_CDS_pos'], ptc_clean[df_col],
                           alpha=0.3, s=15, color='blue', label='PTC TCGA')
    ax_loess_after.plot(ptc_smooth[:, 0], ptc_smooth[:, 1],
                        color='darkblue', linewidth=3, label='PTC LOESS fit')
    ax_loess_after.scatter(dms_norm_clean['PTCposition_nt'], dms_norm_clean['NMDeff_Norm'],
                           alpha=0.3, s=15, color='green', label='DMS (normalized)')
    ax_loess_after.plot(dms_norm_smooth[:, 0], dms_norm_smooth[:, 1],
                        color='darkgreen', linewidth=3, linestyle='--',
                        label='DMS LOESS fit (normalized)')
    ax_loess_after.set_xlabel('PTC Position (nt)')
    ax_loess_after.set_ylabel('NMDeff')
    ax_loess_after.set_title('LOESS Fits After Position-Dependent Normalization')
    ax_loess_after.legend()
    ax_loess_after.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved DMS normalization comparison plots to {output_path}")
    logger.info(f"PTC data: {len(ptc_clean)} variants")
    logger.info(f"DMS data: {len(dms_clean)} variants")


def plot_dms_le_normalization_comparison(ptc_data, dms_data, output_path, df_col="NMDeff"):
    """
    Plot comparison of PTC dataset vs normalized DMS_LE dataset with KDE and violin plots.
    
    Args:
        ptc_data (pd.DataFrame): PTC dataset with NMDeff 
        dms_data (pd.DataFrame): DMS_LE dataset with normalized fitness values and exon_length
        output_path (str): Path to save the plot
        dataset_name (str): Name of the DMS dataset for labeling
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'LE Normalization Comparison', fontsize=16)
    
    # Left plot: KDE comparison of distributions
    sns.kdeplot(ptc_data[df_col].dropna(), 
                label='ASE (PTC dataset)', color='blue', ax=axes[0])
    sns.kdeplot(dms_data['NMDeff_Norm'], 
                label=f'LE (mapped to ASE scale)', color='red', ax=axes[0])
    
    axes[0].set_xlabel('ASE Scale')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Fitness Values Mapped to ASE Scale')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Violin plot showing distribution by exon length
    sns.violinplot(data=dms_data, x='exon_length', y='NMDeff_Norm', palette='Set2', ax=axes[1])
    axes[1].set_title('Distribution of Normalized Fitness by Exon Length')
    axes[1].set_xlabel('Exon Length')
    axes[1].set_ylabel('Normalized Fitness')
    axes[1].grid(alpha=0.3)
    
    # Rotate x-axis labels if needed
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved DMS LE normalization comparison plots to {output_path}")


def plot_dms_pe_normalization_comparison(ptc_data, dms_data, output_path, df_col="NMDeff"):
    """
    Plot comparison of PTC dataset vs normalized DMS_PE dataset with LOESS fits and distributions.
    Similar to DMS_SP plotting but adapted for PE data (distance from EJC).
    
    Args:
        dms_data (pd.DataFrame): DMS_PE dataset with NMDeff, NMDeff_Norm, and nt_position
        ptc_data (pd.DataFrame): PTC dataset with NMDeff and PTC_EJC_dist
        output_path (str): Path to save the plot (optional, will auto-generate if None)
        dataset_name (str): Name of the DMS dataset for labeling
    """

    
    def loess_smooth(x, y, frac=0.3, degree=2):
        """Apply LOESS-like smoothing to data"""
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        n = len(x)
        window_size = max(int(n * frac), 10)
        
        x_smooth = []
        y_smooth = []
        
        for i in range(n):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n, i + window_size // 2)
            
            x_local = x_sorted[start_idx:end_idx]
            y_local = y_sorted[start_idx:end_idx]
            
            if len(x_local) > degree:
                poly_features = PolynomialFeatures(degree=degree)
                poly_reg = Pipeline([
                    ('poly', poly_features),
                    ('linear', LinearRegression())
                ])
                poly_reg.fit(x_local.reshape(-1, 1), y_local)
                y_pred = poly_reg.predict([[x_sorted[i]]])[0]
            else:
                y_pred = np.mean(y_local)
            
            x_smooth.append(x_sorted[i])
            y_smooth.append(y_pred)
        
        return np.array(x_smooth), np.array(y_smooth)
    
    # Auto-generate output path if not provided
    if output_path is None:
        from NMD.config import FIGURES_DIR
        import os
        plot_output_dir = FIGURES_DIR / "data" / "DMS"
        os.makedirs(plot_output_dir, exist_ok=True)
        output_path = plot_output_dir / f"PE_normalization_comparison.png"
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Distribution comparison (before and after)
    ax_dist_before = fig.add_subplot(gs[0, 0])
    ax_dist_after = fig.add_subplot(gs[0, 1])
    
    # Bottom row: LOESS fits (before and after)
    ax_loess_before = fig.add_subplot(gs[1, 0])
    ax_loess_after = fig.add_subplot(gs[1, 1])
    
    # === Top Left: Distribution Before Normalization ===
    ax_dist_before.hist(dms_data['NMDeff'].dropna(), bins=50, alpha=0.6, density=True,
                        color='orange', label='DMS PE (original)')
    ax_dist_before.hist(ptc_data[df_col].dropna(), bins=50, alpha=0.6, density=True,
                        color='blue', label='PTC')
    sns.kdeplot(data=dms_data, x='NMDeff', color='darkorange',
                linewidth=2, ax=ax_dist_before, label='DMS PE KDE')
    sns.kdeplot(data=ptc_data, x=df_col, color='darkblue',
                linewidth=2, ax=ax_dist_before, label='PTC KDE')
    ax_dist_before.set_xlabel('NMDeff')
    ax_dist_before.set_ylabel('Density')
    ax_dist_before.set_title('Distribution Before Normalization')
    ax_dist_before.legend()
    ax_dist_before.grid(True, alpha=0.3)
    
    # === Top Right: Distribution After Normalization ===
    ax_dist_after.hist(dms_data['NMDeff_Norm'].dropna(), bins=50, alpha=0.6, density=True,
                       color='green', label='DMS PE (normalized)')
    ax_dist_after.hist(ptc_data[df_col].dropna(), bins=50, alpha=0.6, density=True,
                       color='blue', label='PTC')
    dms_temp = dms_data.copy()
    dms_temp['NMDeff'] = dms_temp['NMDeff_Norm']
    sns.kdeplot(data=dms_temp, x='NMDeff', color='darkgreen',
                linewidth=2, ax=ax_dist_after, label='DMS PE KDE')
    sns.kdeplot(data=ptc_data, x=df_col, color='darkblue',
                linewidth=2, ax=ax_dist_after, label='PTC KDE')
    ax_dist_after.set_xlabel('NMDeff')
    ax_dist_after.set_ylabel('Density')
    ax_dist_after.set_title('Distribution After Position-Dependent Normalization')
    ax_dist_after.legend()
    ax_dist_after.grid(True, alpha=0.3)
    
    # === Bottom Left: LOESS Fits Before Normalization ===
    # Fit LOESS to PTC data
    ptc_clean = ptc_data.dropna(subset=['PTC_EJC_dist', df_col])
    ptc_x_smooth, ptc_y_smooth = loess_smooth(
        ptc_clean['PTC_EJC_dist'].values,
        ptc_clean[df_col].values,
        frac=0.4
    )
    
    # Fit LOESS to DMS original data
    dms_clean = dms_data.dropna(subset=['nt_position', 'NMDeff'])
    dms_x_smooth, dms_y_smooth = loess_smooth(
        dms_clean['nt_position'].values,
        dms_clean['NMDeff'].values,
        frac=0.4
    )
    
    ax_loess_before.scatter(ptc_clean['PTC_EJC_dist'], ptc_clean['NMDeff'],
                            alpha=0.3, s=15, color='blue', label='PTC TCGA')
    ax_loess_before.plot(ptc_x_smooth, ptc_y_smooth,
                         color='darkblue', linewidth=3, label='PTC LOESS fit')
    ax_loess_before.scatter(dms_clean['nt_position'], dms_clean['NMDeff'],
                            alpha=0.3, s=15, color='orange', label='DMS PE (original)')
    ax_loess_before.plot(dms_x_smooth, dms_y_smooth,
                         color='darkorange', linewidth=3, linestyle='--',
                         label='DMS PE LOESS fit (original)')
    ax_loess_before.set_xlabel('Distance from EJC (nt)')
    ax_loess_before.set_ylabel('NMDeff')
    ax_loess_before.set_title('LOESS Fits Before Normalization')
    ax_loess_before.legend()
    ax_loess_before.grid(True, alpha=0.3)
    
    # === Bottom Right: LOESS Fits After Normalization ===
    # Fit LOESS to normalized DMS data
    dms_norm_clean = dms_data.dropna(subset=['nt_position', 'NMDeff_Norm'])
    dms_norm_x_smooth, dms_norm_y_smooth = loess_smooth(
        dms_norm_clean['nt_position'].values,
        dms_norm_clean['NMDeff_Norm'].values,
        frac=0.4
    )
    
    ax_loess_after.scatter(ptc_clean['PTC_EJC_dist'], ptc_clean[df_col],
                           alpha=0.3, s=15, color='blue', label='PTC TCGA')
    ax_loess_after.plot(ptc_x_smooth, ptc_y_smooth,
                        color='darkblue', linewidth=3, label='PTC LOESS fit')
    ax_loess_after.scatter(dms_norm_clean['nt_position'], dms_norm_clean['NMDeff_Norm'],
                           alpha=0.3, s=15, color='green', label='DMS PE (normalized)')
    ax_loess_after.plot(dms_norm_x_smooth, dms_norm_y_smooth,
                        color='darkgreen', linewidth=3, linestyle='--',
                        label='DMS PE LOESS fit (normalized)')
    ax_loess_after.set_xlabel('Distance from EJC (nt)')
    ax_loess_after.set_ylabel('NMDeff')
    ax_loess_after.set_title('LOESS Fits After Position-Dependent Normalization')
    ax_loess_after.legend()
    ax_loess_after.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved DMS PE normalization comparison plots to {output_path}")
    logger.info(f"PTC data: {len(ptc_clean)} variants")
    logger.info(f"DMS PE data: {len(dms_clean)} variants")


def plot_preprocessing_steps(preprocessing_df, output_path):
    """
    Plot preprocessing steps showing variant counts and NMDeff values.
    
    Creates a dual-axis plot:
    - Left axis: Stacked bar chart showing SNV and indel counts at each step
    - Right axis: Line plots showing mean NMDeff for last exon and triggering groups with CI, split by variant type
    
    Args:
        preprocessing_df (pd.DataFrame): DataFrame with preprocessing statistics
        output_path (str): Path to save the plot
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    col1 = "#ff9e9d"
    col2 = "#022778"
    
    # Set up the x-axis
    x_pos = np.arange(len(preprocessing_df))
    step_names = preprocessing_df['step'].values
    
    # Left axis: Stacked bar chart for variant counts
    width = 0.6
    snv_counts = preprocessing_df['n_snvs'].values
    indel_counts = preprocessing_df['n_indels'].values
    total_counts = snv_counts + indel_counts
    
    # Create stacked bars
    bars_snv = ax1.bar(x_pos, snv_counts, width, label='SNVs', color=col2, alpha=0.8)
    bars_indel = ax1.bar(x_pos, indel_counts, width, bottom=snv_counts, label='Indels', color=col1, alpha=0.8)
    
    # Add total variant count on top of each bar
    for i, (x, total) in enumerate(zip(x_pos, total_counts)):
        ax1.text(x, total + ax1.get_ylim()[1] * 0.02, f'{int(total):,}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Preprocessing Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Variants', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(step_names, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right axis: Line plots for NMDeff with CI, split by variant type
    ax2 = ax1.twinx()
    
    # Define colors matching the bar chart
    snv_color = col2  # Blue for SNVs
    indel_color = col1  # Orange for indels
    
    # Plot last exon (evading) - SNVs
    last_exon_snv_means = preprocessing_df['last_exon_snv_mean'].values
    last_exon_snv_ci_lower = preprocessing_df['last_exon_snv_ci_lower'].values
    last_exon_snv_ci_upper = preprocessing_df['last_exon_snv_ci_upper'].values
    
    ax2.plot(x_pos, last_exon_snv_means, 'o-', color=snv_color, linewidth=2.5, 
             markersize=8, label='Last Exon SNVs', zorder=10, alpha=0.9)
    ax2.fill_between(x_pos, last_exon_snv_ci_lower, last_exon_snv_ci_upper, 
                     alpha=0.2, color=snv_color, zorder=5)
    
    # Plot last exon (evading) - Indels
    last_exon_indel_means = preprocessing_df['last_exon_indel_mean'].values
    last_exon_indel_ci_lower = preprocessing_df['last_exon_indel_ci_lower'].values
    last_exon_indel_ci_upper = preprocessing_df['last_exon_indel_ci_upper'].values
    
    ax2.plot(x_pos, last_exon_indel_means, 'o--', color=indel_color, linewidth=2.5, 
             markersize=8, label='Last Exon Indels', zorder=10, alpha=0.9)
    ax2.fill_between(x_pos, last_exon_indel_ci_lower, last_exon_indel_ci_upper, 
                     alpha=0.2, color=indel_color, zorder=5)
    
    # Plot triggering - SNVs
    triggering_snv_means = preprocessing_df['triggering_snv_mean'].values
    triggering_snv_ci_lower = preprocessing_df['triggering_snv_ci_lower'].values
    triggering_snv_ci_upper = preprocessing_df['triggering_snv_ci_upper'].values
    
    ax2.plot(x_pos, triggering_snv_means, 's-', color=snv_color, linewidth=2.5, 
             markersize=8, label='Triggering SNVs', zorder=10, alpha=0.6)
    ax2.fill_between(x_pos, triggering_snv_ci_lower, triggering_snv_ci_upper, 
                     alpha=0.15, color=snv_color, zorder=5)
    
    # Plot triggering - Indels
    triggering_indel_means = preprocessing_df['triggering_indel_mean'].values
    triggering_indel_ci_lower = preprocessing_df['triggering_indel_ci_lower'].values
    triggering_indel_ci_upper = preprocessing_df['triggering_indel_ci_upper'].values
    
    ax2.plot(x_pos, triggering_indel_means, 's--', color=indel_color, linewidth=2.5, 
             markersize=8, label='Triggering Indels', zorder=10, alpha=0.6)
    ax2.fill_between(x_pos, triggering_indel_ci_lower, triggering_indel_ci_upper, 
                     alpha=0.15, color=indel_color, zorder=5)
    
    ax2.set_ylabel('Mean NMD Efficiency', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Combine legends and place in bottom left
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10, framealpha=0.95)
    
    plt.title('Effect of Preprocessing on PTC Dataset', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved preprocessing steps plot to {output_path}")


def plot_germline_somatic_overlap_comparison(germline_df, somatic_df, overlap_mask, output_path, dataset_name="germline"):
    """
    Plot comparison of NMD efficiency between germline and somatic overlapping variants.
    
    Args:
        germline_df: DataFrame with germline variants (with variant_id column)
        somatic_df: DataFrame with somatic variants (with variant_id column)
        overlap_mask: Boolean mask indicating overlapping variants in germline_df
        output_path: Path to save the plot
        dataset_name: Name of the germline dataset (e.g., "GTEx", "germline_TCGA")
    """
    # Get overlapping variants
    overlapping_germline = germline_df[overlap_mask].copy()
    
    if len(overlapping_germline) == 0:
        logger.warning(f"No overlapping variants to plot for {dataset_name}")
        return
    
    # Create a merged dataframe matching variants by variant_id
    merged = overlapping_germline.merge(
        somatic_df[['variant_id', 'ASE_NMD_efficiency_TPM']], 
        on='variant_id', 
        suffixes=('_germline', '_somatic')
    )
    
    # Use NMDeff if available, otherwise use ASE_NMD_efficiency_TPM
    germline_col = 'NMDeff' if 'NMDeff' in merged.columns else 'ASE_NMD_efficiency_TPM_germline'
    
    if len(merged) == 0:
        logger.warning(f"No matched variants to plot for {dataset_name}")
        return
    
    # Aggregate multiple germline values per variant (take median)
    # Group by variant_id and somatic value (which should be unique per variant_id)
    # and take median of germline values
    n_before = len(merged)
    merged = merged.groupby(['variant_id', 'ASE_NMD_efficiency_TPM_somatic']).agg({
        germline_col: 'median'
    }).reset_index()
    n_after = len(merged)
    
    if n_before != n_after:
        logger.info(f"Aggregated {n_before} germline measurements to {n_after} unique variants (median)")
    
    if len(merged) == 0:
        logger.warning(f"No variants remaining after aggregation for {dataset_name}")
        return
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Scatterplot with correlation
    ax1 = axes[0]
    ax1.scatter(merged['ASE_NMD_efficiency_TPM_somatic'], merged[germline_col], 
                alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    # Add diagonal line
    min_val = min(merged['ASE_NMD_efficiency_TPM_somatic'].min(), merged[germline_col].min())
    max_val = max(merged['ASE_NMD_efficiency_TPM_somatic'].max(), merged[germline_col].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='y=x')
    
    # Calculate correlation
    correlation = np.corrcoef(merged['ASE_NMD_efficiency_TPM_somatic'], merged[germline_col])[0, 1]
    r2 = r2_score(merged['ASE_NMD_efficiency_TPM_somatic'], merged[germline_col])
    
    ax1.set_xlabel('NMD Efficiency (Somatic)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'NMD Efficiency ({dataset_name})', fontsize=12, fontweight='bold')
    ax1.set_title(f'Overlapping Variants: {dataset_name} vs Somatic\n' + 
                  f'Pearson r = {correlation:.3f}, R² = {r2:.3f}, n = {len(merged)}',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Histogram of differences
    ax2 = axes[1]
    differences = merged[germline_col] - merged['ASE_NMD_efficiency_TPM_somatic']
    
    ax2.hist(differences, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(x=differences.mean(), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean = {differences.mean():.3f}')
    
    ax2.set_xlabel(f'NMD Efficiency Difference ({dataset_name} - Somatic)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of Differences\nMean ± SD = {differences.mean():.3f} ± {differences.std():.3f}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved germline-somatic overlap comparison plot to {output_path}")


def plot_transcript_ptc_predictions(
    transcript_id: str,
    gene_name: str,
    ptc_positions: list,
    predictions: list,
    sigma: float = 3.0,
    show_exon_boundaries: bool = True,
    show_55nt_rule: bool = True,
    figsize: tuple = (15, 8),
    gencode_version: str = GENCODE_VERSION
):
    """
    Plot transcript-wide PTC predictions with exon boundaries.
    
    Args:
        transcript_id: Transcript ID (e.g., "ENST00000357654.9")
        gene_name: Gene name for the plot title
        ptc_positions: List of PTC positions in transcript coordinates
        predictions: List of predicted NMD efficiency values
        sigma: Gaussian smoothing sigma (default: 3.0)
        show_exon_boundaries: Whether to show exon boundaries (default: True)
        show_55nt_rule: Whether to show 55nt rule boundary (default: True)
        figsize: Figure size (default: (15, 8))
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply smoothing to the predictions
    smoothed_predictions = gaussian_filter1d(predictions, sigma=sigma)
    
    # Plot smoothed model predictions
    ax.plot(ptc_positions, smoothed_predictions, color='#3d405b', marker='o', 
            markersize=4, linewidth=2, 
            label=f'Smoothed Model Predictions (σ={sigma})', alpha=0.8)
    
    # Get transcript information to calculate exon boundaries
    genome = Genome(gencode_version)
    transcript = genome.transcripts[transcript_id]
    
    exon_boundaries = []
    penultimate_junction = None
    
    if transcript is not None and hasattr(transcript, 'exons') and hasattr(transcript, 'cdss'):
        # Calculate exon boundaries in transcript nucleotide coordinates
        utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
        
        # Get exon boundaries within CDS
        cumulative_length = 0
        
        for exon in transcript.exons:
            cumulative_length += len(exon)
            # Convert to transcript coordinates (matching varpos calculation)
            if cumulative_length > utr5_length:
                # This boundary is within or after CDS
                transcript_boundary = cumulative_length
                if transcript_boundary <= max(ptc_positions):
                    exon_boundaries.append(transcript_boundary)
        
        if show_exon_boundaries:
            # Plot exon boundaries
            for i, boundary in enumerate(exon_boundaries[:-1]):  # Exclude last boundary
                ax.axvline(x=boundary, color='#BDC3C7', linestyle='--', alpha=0.7, linewidth=1)
                if i == 0:
                    ax.axvline(x=boundary, color='#BDC3C7', linestyle='--', alpha=0.7, 
                              linewidth=1, label='Exon boundaries')
        
        # Highlight the penultimate exon junction (last exon junction)
        if len(exon_boundaries) >= 2:
            penultimate_junction = exon_boundaries[-1]  # Last exon junction is the last boundary
            if penultimate_junction <= max(ptc_positions):
                ax.axvline(x=penultimate_junction, color='#E74C3C', linestyle='-', 
                          alpha=0.8, linewidth=2, label='Last exon junction')
        
        # Annotate 55 nt before last exon junction
        if show_55nt_rule and penultimate_junction is not None:
            boundary_55nt = penultimate_junction - 55
            ax.axvline(x=boundary_55nt, color="#CC695E", linestyle='--', alpha=0.8, 
                      linewidth=2, label='55 nt rule boundary')
    
    # Customize the plot
    ax.set_xlabel('PTC Position (Nucleotide in Transcript)', fontsize=12)
    ax.set_ylabel('Predicted NMD Efficiency', fontsize=12)
    title = f'PTC Predictions for {gene_name}' if gene_name else f'PTC Predictions'
    title += f' (Transcript: {transcript_id})'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, color='#ECF0F1')
    ax.legend(fontsize=11)
    ax.set_ylim(-1, 1)
    
    # Add some statistics as text
    stats_text = f'Total PTCs: {len(ptc_positions)}\n'
    stats_text += f'Mean Pred: {np.mean(predictions):.3f}\n'
    stats_text += f'Std Pred: {np.std(predictions):.3f}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='#F8F9FA', alpha=0.9, edgecolor='#BDC3C7'))
    
    plt.tight_layout()
    
    return fig

@app.command()
def plot_dms_sp_gene_predictions(
    dms_file=PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv",
    model_path=MODELS_DIR / "NMDetectiveAI.pt",
    output_dir=FIGURES_DIR / "SP",
    loess_frac=0.3,
    dpi=300
):
    """
    Create individual prediction plots for each gene in DMS SP dataset.
    
    For each gene in the DMS Start Proximal dataset:
    - Generate NMDetectiveAI predictions for all PTCs
    - Plot observed DMS data (scatter + LOESS smoothing)
    - Plot NMDetectiveAI predictions (line)
    - Save to output_dir/{gene}_predictions.png
    
    Args:
        dms_file: Path to DMS SP fitness data
        model_path: Path to NMDetectiveAI model
        output_dir: Directory to save plots
        loess_frac: Fraction for LOESS smoothing
        dpi: Resolution for saved figures
    """
    from NMD.modeling.predict import predict_transcript_ptcs
    from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
    from NMD.modeling.TrainerConfig import TrainerConfig
    from scipy.stats import spearmanr
    import pickle
    from torch.utils.data import DataLoader
    from NMD.modeling.SequenceDataset import SequenceDataset
    
    # Ensure numeric parameters are actually numeric
    loess_frac = float(loess_frac)
    dpi = int(dpi)
    
    logger.info(f"Loading DMS SP data from {dms_file}")
    dms_data = pd.read_csv(dms_file)
    # Prepare output table: copy of original DMS fitness table with prediction columns
    dms_data_out = dms_data.copy()
    dms_data_out['NMDetectiveAI_pred_genome_kit'] = np.nan
    dms_data_out['NMDetectiveAI_pred_dms_seq'] = np.nan
    
    # Load pre-processed DMS SP sequences
    sequences_file = PROCESSED_DATA_DIR / "DMS_SP" / "processed_sequences.pkl"
    logger.info(f"Loading pre-processed sequences from {sequences_file}")
    with open(sequences_file, 'rb') as f:
        dms_sequences = pickle.load(f)
    logger.info(f"Loaded {len(dms_sequences)} pre-processed sequences")
    
    # Get unique genes
    genes = dms_data['gene'].unique()
    logger.info(f"Found {len(genes)} genes in DMS SP dataset: {', '.join(genes)}")
    
    # Load gene symbol to Ensembl ID mapping
    annotation_file = PROCESSED_DATA_DIR.parent / "raw" / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
    logger.info(f"Loading gene annotations from {annotation_file}")
    annotations = pd.read_csv(annotation_file, sep='\t')
    gene_symbol_to_id = dict(zip(annotations['gene_name'], annotations['gene_id']))
    logger.info(f"Loaded {len(gene_symbol_to_id)} gene symbol to Ensembl ID mappings")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NMDetectiveAI model once
    logger.info(f"Loading NMDetectiveAI model from {model_path}")
    config = TrainerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    load_model(model, str(model_path), device=device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    for gene in genes:
        logger.info(f"\nProcessing {gene}...")
        
        # Get Ensembl gene ID from symbol
        if gene not in gene_symbol_to_id:
            logger.warning(f"Gene symbol {gene} not found in annotations, skipping...")
            continue
        
        ensembl_gene_id = gene_symbol_to_id[gene]
        logger.info(f"  Gene symbol {gene} -> Ensembl ID {ensembl_gene_id}")
        
        # Find first transcript with CDS for this gene
        transcript = None
        transcript_id = None
        
        try:
            gene_obj = genome.genes[ensembl_gene_id]
            for tx in gene_obj.transcripts:
                if tx.cdss and len(tx.cdss) > 0:
                    transcript = tx
                    transcript_id = tx.id
                    logger.info(f"  Using transcript {transcript_id}")
                    break
        except KeyError:
            logger.warning(f"Ensembl gene ID {ensembl_gene_id} not found in genome, skipping...")
            continue
        
        if transcript is None:
            logger.warning(f"No transcript with CDS found for {ensembl_gene_id}, skipping...")
            continue
        
        # Get 5'UTR length for coordinate conversion
        utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
        
        # Get DMS data for this gene
        gene_dms = dms_data[dms_data['gene'] == gene].copy()
        gene_dms = gene_dms.dropna(subset=['PTCposition_nt', 'NMDeff'])
        gene_dms = gene_dms.sort_values('PTCposition_nt')
        
        if len(gene_dms) == 0:
            logger.warning(f"No valid DMS data for {gene}, skipping...")
            continue
        
        logger.info(f"  Found {len(gene_dms)} DMS observations")
        
        # Generate predictions for first 80 PTCs only
        logger.info(f"  Generating predictions for first 80 PTCs...")
        results = predict_transcript_ptcs(
            gene_name=gene,
            transcript_id=transcript_id,
            transcript_idx=0,
            model_path=str(model_path),
            max_positions=84
        )
        
        ptc_positions = results['ptc_positions']
        predictions = results['predictions']
        
        # Convert positions to CDS coordinates and ensure they're numeric arrays
        cds_positions = np.array([float(pos - utr5_length) for pos in ptc_positions])
        predictions = np.array([float(p) for p in predictions])
        
        logger.info(f"  Generated {len(predictions)} predictions")
        
        # Get pre-processed sequences for this gene and predict
        logger.info(f"  Predicting on pre-processed DMS SP sequences...")
        
        # Get DMS data for this gene and sort it consistently
        gene_dms_full = dms_data[dms_data['gene'] == gene].copy()
        gene_dms_full = gene_dms_full.dropna(subset=['PTCposition_nt', 'NMDeff'])
        gene_dms_full = gene_dms_full.sort_values('PTCposition_nt')
        
        # Get sequences in the same sorted order
        gene_indices = gene_dms_full.index.tolist()
        gene_sequences = [dms_sequences[i] for i in gene_indices]
        
        # Create dataset and dataloader for DMS sequences
        eval_dataset = SequenceDataset(gene_dms_full, gene_sequences, label_col='NMDeff')
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        
        # Run predictions on DMS sequences
        dms_predictions = []
        with torch.no_grad():
            for batch_sequences, batch_lengths, _ in eval_loader:
                batch_sequences, batch_lengths = [
                    x.to(device) for x in (batch_sequences, batch_lengths)
                ]
                batch_preds = model(batch_sequences, batch_lengths).squeeze()
                
                # Handle single prediction case
                if batch_preds.dim() == 0:
                    dms_predictions.append(float(batch_preds.cpu().numpy()))
                else:
                    dms_predictions.extend(batch_preds.cpu().numpy())
        
        dms_predictions = np.array([float(p) for p in dms_predictions])
        logger.info(f"  Generated {len(dms_predictions)} predictions on DMS sequences")
        
        # Use raw DMS observations (no scaling) - ensure they're numeric
        dms_positions = pd.to_numeric(gene_dms['PTCposition_nt'], errors='coerce').values.astype(float)
        dms_values = pd.to_numeric(gene_dms['NMDeff'], errors='coerce').values.astype(float)
        
        # Remove any NaN values that resulted from conversion
        valid_mask = ~(np.isnan(dms_positions) | np.isnan(dms_values))
        dms_positions = dms_positions[valid_mask]
        dms_values = dms_values[valid_mask]
        
        if len(dms_positions) == 0:
            logger.warning(f"No valid numeric DMS data for {gene} after filtering, skipping...")
            continue
        
        # Calculate Spearman correlation between predictions and observations
        # Match predictions to DMS positions by interpolating
        pred_dict = {int(pos): float(pred) for pos, pred in zip(cds_positions, predictions)}
        matched_preds = []
        matched_obs = []
        for pos, obs in zip(dms_positions, dms_values):
            if int(pos) in pred_dict:
                matched_preds.append(float(pred_dict[int(pos)]))
                matched_obs.append(float(obs))
        
        spearman_r = None
        if len(matched_preds) > 0:
            spearman_r, _ = spearmanr(matched_obs, matched_preds)
        
        # Create plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # First y-axis for predictions
        # Plot genome-kit based predictions (dark blue)
        ax1.plot(cds_positions.astype(float), predictions.astype(float), 
               color='#022778', linewidth=2.5, 
               label='NMDetective-AI (genome-kit PTCs)', zorder=3)
        
        # Plot DMS sequence-based predictions (light blue)
        ax1.scatter(dms_positions.astype(float), dms_predictions.astype(float),
                   color='#6ac3e9', alpha=0.6, s=40, marker='o',
                   label='NMDetective-AI (DMS sequences)', zorder=4)
        
        ax1.set_xlabel('PTC Position in CDS (nt)', fontsize=18)
        ax1.set_ylabel('Predicted NMD efficiency', fontsize=18, color='#022778')
        ax1.tick_params(axis='y', labelcolor='#022778', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Second y-axis for DMS observations
        ax2 = ax1.twinx()
        
        # Plot DMS observations (scatter)
        ax2.scatter(dms_positions.astype(float), dms_values.astype(float),
                  color='#ff9e9d', alpha=0.4, s=30,
                  label='DMS observations', zorder=1)
        
        # Plot LOESS smoothed DMS trend
        # Data is already numeric from earlier conversion, just sort for LOESS
        sort_idx = np.argsort(dms_positions)
        dms_pos_sorted = dms_positions[sort_idx]
        dms_val_sorted = dms_values[sort_idx]
        
        dms_smooth = lowess(dms_val_sorted, dms_pos_sorted, frac=loess_frac)
        ax2.plot(dms_smooth[:, 0], dms_smooth[:, 1],
                color='#ffdfcb', linewidth=2.5,
                label='DMS (LOESS)', zorder=2)
        
        ax2.set_ylabel('Observed NMD efficiency (DMS)', fontsize=18, color='#ff9e9d')
        ax2.tick_params(axis='y', labelcolor='#ff9e9d', labelsize=14)
        
        # Add gene name and correlation as title
        title = f'{gene}'
        if spearman_r is not None:
            title += f' (Spearman R = {spearman_r:.3f})'
        ax1.set_title(title, fontsize=20, fontweight='bold')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=13, loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"{gene}_predictions.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"  Saved plot to {output_path}")
        plt.close(fig)
        # ------------------ Save predictions into table ------------------
        # Fill genome-kit based predictions by matching integer PTC positions
        # pred_dict maps integer positions (CDS coords) to prediction values
        for idx, row in gene_dms.iterrows():
            try:
                key = int(row['PTCposition_nt'])
            except Exception:
                key = None
            if key is not None and key in pred_dict:
                dms_data_out.loc[idx, 'NMDetectiveAI_pred_genome_kit'] = float(pred_dict[key])

        # Fill DMS sequence-based predictions (aligned by gene_indices)
        for i, gi in enumerate(gene_indices):
            if i < len(dms_predictions):
                dms_data_out.loc[gi, 'NMDetectiveAI_pred_dms_seq'] = float(dms_predictions[i])
        # ----------------------------------------------------------------
    
    # Save annotated fitness table with predictions
    out_dir = TABLES_DIR / "SP"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fitness_with_predictions.csv"
    try:
        dms_data_out.to_csv(out_file, index=False)
        logger.info(f"Saved annotated fitness table with predictions to {out_file}")
    except Exception as e:
        logger.error(f"Failed to save annotated fitness table: {e}")

    logger.success(f"\nCompleted plotting for {len(genes)} genes")


if __name__ == "__main__":
    app()
