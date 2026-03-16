"""
Model evaluation utilities for NMD prediction models.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
from loguru import logger
import wandb

from NMD.config import PROCESSED_DATA_DIR
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.features import load_data
from NMD.plots import analyse_predictions
from NMD.utils import relative_squared_error, load_model, collate_fn
from NMD.modeling.TrainerConfig import TrainerConfig


class ModelEvaluator:
    """Handles model evaluation on various datasets."""
    
    def __init__(self, model_path: Path, config: TrainerConfig):
        self.model_path = model_path
        self.config = config
        self.out_dir = model_path.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _setup_model(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Setup model and criterion for evaluation."""
        from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
        
        # Create NMDetectiveAI model
        model = NMDetectiveAI(
            hidden_dims=self.config.dnn_hidden_dims,
            dropout=self.config.dnn_dropout,
            random_init=self.config.random_init,
            use_mlm=self.config.Orthrus_MLM,
            activation_function=self.config.activation_function,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.device)
        criterion = torch.nn.MSELoss()
        
        return model, criterion
        
    def evaluate_single_dataset(
        self, 
        data_path: Path, 
    ) -> Dict[str, float]:
        """Evaluate model on a single dataset."""
        # Load data
        sequences, metadata = load_data(data_path)
        col = "NMDeff_Norm" if self.config.normalize else "NMDeff"
        
        dataset = SequenceDataset(metadata, sequences, label_col=col)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # Setup and load model
        model, criterion = self._setup_model()
        load_model(model, self.model_path, device=self.device)
        model.eval()
        
        # Get predictions
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_sequences, batch_lengths, batch_labels in tqdm(dataloader, desc="Evaluating"):
                batch_sequences, batch_lengths, batch_labels = [
                    x.to(self.device) for x in (batch_sequences, batch_lengths, batch_labels)
                ]
                predictions = model(batch_sequences, batch_lengths).squeeze()
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(batch_labels.cpu().numpy().flatten())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, criterion)   
        simple_output_file = self._save_predictions_with_metadata(all_labels, all_preds, data_path.name, metadata, col)
        analyse_predictions(simple_output_file)

        return metrics
    
    def _save_predictions_with_metadata(
        self, 
        labels: List[float], 
        predictions: List[float], 
        dataset_name: str,
        metadata: pd.DataFrame,
        label_col: str
    ) -> pd.DataFrame:
        """Save predictions with full metadata and generate plots."""
        # Create predictions dataframe with metadata
        predictions_df = metadata.copy()
        predictions_df["predictions"] = predictions
        
        # Save with metadata
        output_file = self.out_dir / f"{dataset_name}_test_predictions_with_metadata.csv"
        predictions_df.to_csv(output_file, index=False)
        
        # Also save simple format for plotting compatibility
        simple_df = pd.DataFrame({"NMDeff": labels, "predictions": predictions})
        simple_output_file = self.out_dir / f"{dataset_name}_PTC_test_predictions.csv"
        simple_df.to_csv(simple_output_file, index=False)
        
        return simple_output_file

    def _calculate_metrics(
        self, 
        labels: List[float], 
        predictions: List[float],
        criterion: torch.nn.Module
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        test_loss = criterion(torch.tensor(predictions), torch.tensor(labels)).item()
        test_r2 = r2_score(labels, predictions)
        test_corr = np.corrcoef(labels, predictions)[0, 1]
        test_rse = relative_squared_error(labels, predictions)
        
        # Binary classification metrics
        labels_binary = np.array(labels) > 0
        preds_binary = np.array(predictions) > 0
        test_auc = roc_auc_score(labels_binary, preds_binary)
        test_ap = average_precision_score(labels_binary, preds_binary)
        
        return {
            "loss": test_loss,
            "r2": test_r2,
            "correlation": test_corr,
            "rse": test_rse,
            "auc": test_auc,
            "average_precision": test_ap
        }
    
    def _calculate_subset_metrics(self, predictions_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Calculate metrics for dataset subsets based on available categorical columns."""
        
        # Determine which type of subsets to use
        # Check if this is a combined dataset by looking for the categorical columns
        is_combined = ('dataset' in predictions_df.columns and 
                      'var_type' in predictions_df.columns and 
                      'data_type' in predictions_df.columns)
        
        if is_combined:
            # For combined dataset, use dataset, var_type, data_type
            subset_cols = ['dataset', 'var_type', 'data_type']
            logger.info("Using combined dataset subsets: dataset, var_type, data_type")
        else:
            # For PTC datasets, use binary rule columns
            subset_cols = ['Last_Exon', 'Start_Prox', 'Long_Exon', 'Penultimate_Exon']
            logger.info("Using PTC dataset subsets: Last_Exon, Start_Prox, Long_Exon, Penultimate_Exon")
        
        results = []
        
        # Full dataset metrics
        full_metrics = self._calculate_single_subset_metrics(
            predictions_df[label_col], 
            predictions_df['predictions']
        )
        full_metrics['subset'] = 'Full Dataset'
        results.append(full_metrics)
        
        # Calculate metrics for each subset
        for col in subset_cols:
            if col in predictions_df.columns:
                # For categorical columns (dataset, var_type, data_type), iterate over unique values
                if col in ['dataset', 'var_type', 'data_type']:
                    for value in predictions_df[col].unique():
                        subset_data = predictions_df[predictions_df[col] == value]
                        if len(subset_data) > 1:  # Need at least 2 samples for meaningful metrics
                            subset_metrics = self._calculate_single_subset_metrics(
                                subset_data[label_col], 
                                subset_data['predictions']
                            )
                            subset_metrics['subset'] = f"{col}_{value}"
                            results.append(subset_metrics)
                            
                            logger.info(f"Subset {col}={value}: n={len(subset_data)}, R²={subset_metrics['r_squared']:.4f}, Corr={subset_metrics['correlation']:.4f}")
                        else:
                            logger.warning(f"Subset {col}={value} has insufficient data (n={len(subset_data)})")
                else:
                    # For binary columns, check where value == 1
                    subset_data = predictions_df[predictions_df[col] == 1]
                    if len(subset_data) > 1:  # Need at least 2 samples for meaningful metrics
                        subset_metrics = self._calculate_single_subset_metrics(
                            subset_data[label_col], 
                            subset_data['predictions']
                        )
                        subset_metrics['subset'] = col
                        results.append(subset_metrics)
                        
                        logger.info(f"Subset {col}: n={len(subset_data)}, R²={subset_metrics['r_squared']:.4f}, Corr={subset_metrics['correlation']:.4f}")
                    else:
                        logger.warning(f"Subset {col} has insufficient data (n={len(subset_data)})")
            else:
                logger.warning(f"Column {col} not found in dataset")
        
        return pd.DataFrame(results)
    
    def _calculate_single_subset_metrics(self, true_values, predictions) -> Dict[str, float]:
        """Calculate evaluation metrics for a single subset."""
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        correlation = np.corrcoef(true_values, predictions)[0, 1] if len(true_values) > 1 else np.nan
        r_squared = r2_score(true_values, predictions)
        mse = np.mean((true_values - predictions) ** 2)
        rse = relative_squared_error(true_values, predictions)
        
        # Binary classification metrics
        labels_binary = true_values > 0
        preds_binary = predictions > 0
        auc = roc_auc_score(labels_binary, preds_binary) if len(np.unique(labels_binary)) > 1 else np.nan
        ap = average_precision_score(labels_binary, preds_binary) if len(np.unique(labels_binary)) > 1 else np.nan
        
        return {
            'correlation': correlation,
            'r_squared': r_squared,
            'mse': mse,
            'rse': rse,
            'auc': auc,
            'average_precision': ap,
            'n_samples': len(true_values)
        }
    
    def evaluate_all_ptc_datasets(self) -> pd.DataFrame:
        """Evaluate model on all PTC datasets and combine results."""
        datasets = ["somatic_TCGA", "germline_TCGA", "GTEx"]
        all_metrics = []

        for dataset_name in datasets:
            logger.info(f"Evaluating on {dataset_name} PTC data")
            data_path = PROCESSED_DATA_DIR / "PTC" / dataset_name
            
            metrics = self.evaluate_single_dataset(data_path)
            metrics["dataset"] = dataset_name
            all_metrics.append(metrics)
            
            # Log metrics
            logger.info(f"Results - R2: {metrics['r2']:.4f}, Corr: {metrics['correlation']:.4f}")
            
            # Log to wandb if active
            if wandb.run:
                wandb.log({
                    f"{dataset_name}_r2": metrics['r2'],
                    f"{dataset_name}_correlation": metrics['correlation'],
                    f"{dataset_name}_loss": metrics['loss']
                })
        return pd.DataFrame(all_metrics)
    
    def evaluate_dms_dataset(self, data_path: Path, dataset_name: str = "DMS") -> pd.DataFrame:
        """Evaluate model on DMS dataset with gene-level metrics and wandb logging.
        
        Args:
            data_path: Path to DMS dataset directory or fitness.csv file
            dataset_name: Name prefix for wandb logging (e.g., "PE", "LE", "SP")
            
        Returns:
            DataFrame with gene-level metrics
        """
        from NMD.modeling.features import setup_data
        import pickle
        
        # Load data using DMS-specific loading logic
        # If data_path points to fitness.csv, use parent directory
        if data_path.suffix == '.csv':
            data_dir = data_path.parent
        else:
            data_dir = data_path
        
        with open(data_dir / "processed_sequences.pkl", "rb") as f:
            sequences = pickle.load(f)
        metadata = pd.read_csv(data_dir / "fitness.csv")
        
        label_col = "NMDeff_Norm" if self.config.normalize else "NMDeff"
        
        # Setup and load model
        model, _ = self._setup_model()
        load_model(model, self.model_path, device=self.device)
        
        # Evaluate at gene level
        gene_metrics_df, results_df = evaluate_dms_gene_level(
            model, sequences, metadata, label_col, self.device
        )
        
        # Log gene-level metrics
        for _, row in gene_metrics_df.iterrows():
            gene = row['gene']
            logger.info(
                f"{dataset_name} {gene}: Correlation={row['correlation']:.4f}, "
                f"R²={row['r_squared']:.4f}, n={row['n_samples']}"
            )
            
            if wandb.run:
                wandb.log({
                    f"{dataset_name}_{gene}_correlation": row['correlation'],
                    f"{dataset_name}_{gene}_r2": row['r_squared'],
                    f"{dataset_name}_{gene}_n_samples": row['n_samples']
                })
        
        # Overall metrics
        if len(results_df) > 1:
            overall_corr = np.corrcoef(
                results_df[label_col], results_df['predictions']
            )[0, 1]
            overall_r2 = r2_score(results_df[label_col], results_df['predictions'])
            
            logger.info(f"{dataset_name} Overall: Correlation={overall_corr:.4f}, R²={overall_r2:.4f}")
            
            if wandb.run:
                wandb.log({
                    f"{dataset_name}_overall_correlation": overall_corr,
                    f"{dataset_name}_overall_r2": overall_r2
                })
        
        return gene_metrics_df


class DMSEvaluator:
    """Specialized evaluator for DMS datasets with gene-level analysis."""
    
    def __init__(self, model: torch.nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_by_genes(
        self, 
        dataloader: DataLoader, 
        metadata: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate model performance on gene level."""
        self.model.eval()
        
        all_predictions = []
        with torch.no_grad():
            for batch_sequences, batch_lengths, batch_labels in tqdm(dataloader, desc="Testing"):
                batch_sequences, batch_lengths, batch_labels = [
                    x.to(self.device) for x in (batch_sequences, batch_lengths, batch_labels)
                ]
                predictions = self.model(batch_sequences, batch_lengths).squeeze()
                all_predictions.extend(predictions.cpu().numpy().flatten())
        
        # Add predictions to dataframe
        results_df = metadata.copy()
        results_df["predictions"] = all_predictions
        
        # Calculate metrics per gene
        gene_metrics = []
        label_col = ("NMDeff" if not self.config.normalize else "NMDeff_Norm")
        
        for gene in results_df["gene"].unique():
            gene_data = results_df[results_df["gene"] == gene]
            true_values = gene_data[label_col]
            pred_values = gene_data["predictions"]
            
            if len(true_values) > 1:  # Need at least 2 points for correlation
                metrics = {
                    "gene": gene,
                    "n_samples": len(gene_data),
                    "r2": r2_score(true_values, pred_values),
                    "corr": np.corrcoef(true_values, pred_values)[0, 1] if len(true_values) > 1 else np.nan,
                    "mse": ((true_values - pred_values) ** 2).mean(),
                    "rse": relative_squared_error(true_values, pred_values),
                }
                gene_metrics.append(metrics)
        
        gene_metrics_df = pd.DataFrame(gene_metrics)
        
        # Log median metrics to wandb
        if not gene_metrics_df.empty and wandb.run:
            median_metrics = gene_metrics_df[["r2", "corr", "mse", "rse"]].median().to_dict()
            wandb.log({
                "test_genes_median_r2": median_metrics["r2"],
                "test_genes_median_corr": median_metrics["corr"],
                "test_genes_median_mse": median_metrics["mse"],
                "test_genes_median_rse": median_metrics["rse"],
            })
        
        return results_df, gene_metrics_df


def predict_on_dataset(model, sequences, metadata, label_col, device):
    """Generate predictions for a dataset."""
    from NMD.utils import collate_fn
    
    dataset = SequenceDataset(metadata, sequences, label_col=label_col)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_sequences, batch_lengths, batch_labels in tqdm(dataloader, desc="Generating predictions"):
            batch_sequences, batch_lengths, batch_labels = [
                x.to(device) for x in (batch_sequences, batch_lengths, batch_labels)
            ]
            predictions = model(batch_sequences, batch_lengths).squeeze()
            if predictions.dim() == 0:
                all_preds.append(float(predictions.cpu().numpy()))
            else:
                all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
    
    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(true_values, predictions):
    """Calculate evaluation metrics."""
    correlation = np.corrcoef(true_values, predictions)[0, 1]
    r_squared = r2_score(true_values, predictions)
    mse = np.mean((true_values - predictions) ** 2)
    
    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'mse': mse,
        'n_samples': len(true_values)
    }


def evaluate_dms_gene_level(model, sequences, metadata, label_col, device):
    """Evaluate DMS dataset at gene level."""
    predictions, _ = predict_on_dataset(model, sequences, metadata, label_col, device)
    
    results_df = metadata.copy()
    results_df['predictions'] = predictions

    # if no gene column then we only have BRCA1
    if 'gene' not in results_df.columns:
        results_df['gene'] = 'BRCA1'
    
    gene_metrics = []
    for gene in results_df['gene'].unique():
        gene_data = results_df[results_df['gene'] == gene]
        true_values = gene_data[label_col].values
        pred_values = gene_data['predictions'].values
        
        if len(true_values) > 1:
            metrics = calculate_metrics(true_values, pred_values)
            metrics['gene'] = gene
            gene_metrics.append(metrics)
    
    return pd.DataFrame(gene_metrics), results_df


def evaluate_ptc_subsets(predictions_df, label_col='ASE_NMD_efficiency_TPM'):
    """Calculate metrics for PTC dataset subsets.
    
    DEPRECATED: Use ModelEvaluator._calculate_subset_metrics instead.
    This function is kept for backward compatibility.
    """
    subset_cols = ['Last_Exon', 'Penultimate_Exon', 'Start_Prox', 'Long_Exon']
    
    results = []
    
    # Full dataset
    if len(predictions_df) > 1:
        metrics = calculate_metrics(predictions_df[label_col], predictions_df['predictions'])
        metrics['subset'] = 'Full Dataset'
        results.append(metrics)
    
    # Each subset
    for col in subset_cols:
        if col in predictions_df.columns:
            subset_data = predictions_df[predictions_df[col] == 1]
            if len(subset_data) > 1:
                metrics = calculate_metrics(subset_data[label_col], subset_data['predictions'])
                metrics['subset'] = col
                results.append(metrics)
    
    return pd.DataFrame(results)
