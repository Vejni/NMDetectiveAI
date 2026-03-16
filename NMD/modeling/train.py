from pathlib import Path
from typing import Optional
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import typer
from loguru import logger
from genome_kit import Genome

from NMD.config import OUT_DIR, PROCESSED_DATA_DIR, SEED, WANDB_PROJECT, GENCODE_VERSION, VAL_CHRS
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.Trainer import Trainer
from NMD.modeling.evaluation import ModelEvaluator, DMSEvaluator
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.features import setup_data, get_gene_cv_splits
from NMD.utils import collate_fn, copy_preprocessing_configs
from NMD.plots import (
    plot_predictions,
    plot_gene_ptc_fitness,
    plot_gene_metrics_distribution,
    plot_LE_predictions,
    plot_PE_predictions,
)

app = typer.Typer()


def _generate_run_name(train_type: str, pretrained_model_path: Optional[Path] = None) -> str:
    """Generate standardized run names."""
    if pretrained_model_path:
        model_suffix = pretrained_model_path.stem.replace(".pt", "")
        return f"{train_type}_from_{model_suffix}"
    return train_type


def _setup_wandb_and_trainer(
    config: TrainerConfig,
    run_name: str,
    train_type: str,
    mode: str = "",
    init_wandb: bool = True,
) -> Trainer:
    """Setup wandb and trainer with consistent configuration."""
    if init_wandb and not wandb.run:
        wandb.init(project=WANDB_PROJECT, config=config.__dict__, name=run_name)
    return Trainer(config, train_type, mode)


@app.command()
def train_ptc(
    pretrained_model_path: Optional[Path] = None,
    max_eval_steps: int = 500,
    normalize: bool = True,
    input_file: str = "somatic_TCGA",
):
    """Fine-tune Orthrus model for NMD efficiency prediction on PTC/TCGA data."""
    config = TrainerConfig()
    config.normalize = normalize
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.pretrained_model_path = pretrained_model_path

    train_type = "PTC"
    run_name = _generate_run_name(train_type, pretrained_model_path)
    data_path = PROCESSED_DATA_DIR / "PTC" / input_file

    # Setup trainer and data
    trainer = _setup_wandb_and_trainer(config, run_name, train_type, "")
    train_loader, val_loader, _, _ = setup_data(
        data_path, config.batch_size, "PTC", normalize=normalize
    )

    # Train model
    model_path = trainer.train_step_based(train_loader, val_loader)

    # Copy all preprocessing configurations to model output directory
    copy_preprocessing_configs(model_path.parent, PROCESSED_DATA_DIR)

    # Comprehensive evaluation with subset analysis
    evaluator = ModelEvaluator(model_path, config)
    evaluator.evaluate_all_ptc_datasets()
    
    # Evaluate on PE dataset
    logger.info("Evaluating on DMS datasets")
    evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv", dataset_name="PE")
    evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv", dataset_name="LE")
    evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv", dataset_name="SP")

    wandb.finish()
    logger.success(f"PTC training complete. Model saved to {model_path}")


@app.command()
def train_dms_sp(
    minigene_sequences: bool = False,
    normalize: bool = False,
    pretrained_model_path: Optional[Path] = None,
    max_eval_steps: int = 100,
):
    """Train model on DMS SP dataset."""
    config = TrainerConfig()
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.normalize = normalize
    config.pretrained_model_path = pretrained_model_path
    config.learning_rate = 0.0001

    train_type = "DMS_SP"
    mode = "minigene" if minigene_sequences else ""
    if normalize:
        mode = f"{mode}_normalize" if mode else "normalize"

    run_name = _generate_run_name(f"{train_type}_{mode}", pretrained_model_path)

    # Setup trainer and data
    trainer = _setup_wandb_and_trainer(config, run_name, train_type, mode)
    data_path = (
        PROCESSED_DATA_DIR
        / "DMS_SP"
        / ("fitness_minigene.csv" if minigene_sequences else "fitness.csv")
    )
    sequences, metadata = setup_data(
        data_path,
        batch_size=config.batch_size,
        data_type="DMS_minigene" if minigene_sequences else "DMS",
    )

    # Add chromosome information for stratification
    metadata = _add_chromosome_column(metadata)

    # Setup data loaders
    train_loader, val_loader, _, _ = _setup_dms_data_loaders(
        metadata, sequences, config, split_by="chr"
    )

    # Train model
    model_path = trainer.train_step_based(train_loader, val_loader)

    # Generate predictions and plots
    logger.info("Generating predictions for visualization...")

    # Get predictions for both train and validation sets
    train_predictions, train_labels = trainer.get_predictions(train_loader)
    val_predictions, val_labels = trainer.get_predictions(val_loader)

    # Create combined results dataframe with proper set labels
    train_results = pd.DataFrame(
        {"predictions": train_predictions, "NMDeff": train_labels, "set": "train"}
    )
    val_results = pd.DataFrame(
        {"predictions": val_predictions, "NMDeff": val_labels, "set": "validation"}
    )

    # Combine train and validation results
    results_df = pd.concat([train_results, val_results], ignore_index=True)

    # Save predictions
    results_df.to_csv(trainer.output_dir / "predictions.csv", index=False)

    # Generate plots
    plot_predictions(results_df, trainer.output_dir / "predictions_plot.png")

    wandb.finish()
    logger.success(f"DMS SP training complete. Model saved to {model_path}")


@app.command()
def train_dms_sp_cv(
    n_splits: int = 10,
    minigene_sequences: bool = False,
    normalize: bool = True,
    pretrained_model_path: Optional[Path] = None,
    max_eval_steps: int = 100,
):
    """Train model using gene-based cross-validation on DMS SP dataset."""
    config = TrainerConfig()
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.normalize = normalize
    config.pretrained_model_path = pretrained_model_path
    config.learning_rate = 0.0001

    train_type = "DMS_SP"
    mode = "CV" + ("_minigene" if minigene_sequences else "")
    mode = mode + ("_normalize" if normalize else "")
    run_name = _generate_run_name(f"{train_type}_{mode}", pretrained_model_path)

    # Setup data
    data_path = (
        PROCESSED_DATA_DIR
        / "DMS_SP"
        / ("fitness_minigene.csv" if minigene_sequences else "fitness.csv")
    )
    sequences, metadata = setup_data(
        data_path,
        batch_size=config.batch_size,
        data_type="DMS_minigene" if minigene_sequences else "DMS",
    )
    splits = get_gene_cv_splits(metadata, n_splits)

    # Output directory - use main wandb id for CV results
    cv_output_dir = OUT_DIR / train_type / mode
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cv_output_dir / "Gene_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation training
    all_test_results, all_gene_metrics = [], []

    for fold, split in enumerate(splits):
        logger.info(f"Training fold {fold + 1}/{n_splits}")

        # Create train and test dataframes from gene lists
        train_df = metadata[metadata["gene"].isin(split["train_genes"])]
        test_df = metadata[metadata["gene"].isin(split["test_genes"])]

        # Create the split dict with dataframes
        fold_split = {"train_df": train_df, "test_df": test_df}

        # Setup fold-specific trainer
        fold_trainer = _setup_wandb_and_trainer(
            config, f"{run_name}_fold_{fold}", train_type, mode
        )

        # Setup data for this fold
        train_loader, val_loader = _setup_cv_fold_data(fold_split, sequences, config)

        # Train fold
        fold_trainer.train_step_based(train_loader, val_loader)

        # Evaluate on test genes
        test_results, gene_metrics = DMSEvaluator(fold_trainer.model, config).evaluate_by_genes(
            val_loader, test_df
        )

        # Save fold results
        test_results["fold"] = fold
        gene_metrics["fold"] = fold
        all_test_results.append(test_results)
        all_gene_metrics.append(gene_metrics)

        # Plot individual genes for this fold
        for gene in test_results["gene"].unique():
            gene_data = test_results[test_results["gene"] == gene]
            plot_gene_ptc_fitness(
                gene_data,
                gene,
                plots_dir / f"fold_{fold}_gene_{gene}.png",
                fold=fold,
                col=("NMDeff" if not config.normalize else "NMDeff_Norm"),
            )

        wandb.finish()

    # Combine and save results
    combined_results = pd.concat(all_test_results)
    combined_results.to_csv(cv_output_dir / "all_gene_results.csv", index=False)

    combined_metrics = pd.concat(all_gene_metrics)
    combined_metrics.to_csv(cv_output_dir / "all_gene_metrics.csv", index=False)

    # Plot gene metrics distribution
    plot_gene_metrics_distribution(
        combined_metrics, cv_output_dir / "gene_metrics_distribution.png"
    )

    # Log summary statistics
    logger.info(f"Mean R² across all genes: {combined_metrics['r2'].median():.3f}")
    logger.info(f"Mean correlation across all genes: {combined_metrics['corr'].median():.3f}")
    logger.info(f"Mean MSE across all genes: {combined_metrics['mse'].median():.3f}")
    logger.success(f"Cross-validation complete. Results saved to {cv_output_dir}")


@app.command()
def train_dms_le(
    pretrained_model_path: Optional[Path] = None,
    normalize: bool = True,
    max_eval_steps: int = 500,
):
    """Train model on DMS LE (Long Exon) dataset."""
    config = TrainerConfig()
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.normalize = normalize
    config.learning_rate = 0.0001
    config.pretrained_model_path = pretrained_model_path

    train_type = "DMS_LE"
    mode = "normalize" if normalize else ""
    run_name = _generate_run_name(f"{train_type}_{mode}", pretrained_model_path)

    # Setup trainer and data
    trainer = _setup_wandb_and_trainer(config, run_name, train_type, mode)
    data_path = PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv"
    sequences, metadata = setup_data(data_path, batch_size=config.batch_size, data_type="DMS")

    # Setup data loaders with stratification by sublib
    train_loader, val_loader, train_df, val_df = _setup_dms_data_loaders(
        metadata, sequences, config, stratify_by="sublib"
    )

    # Train and evaluate
    model_path = trainer.train_step_based(train_loader, val_loader)

    # Generate predictions and plots for LE dataset
    logger.info("Generating predictions for visualization...")

    # Create non-shuffled data loaders for predictions to preserve order
    label_col = "NMDeff" if not config.normalize else "NMDeff_Norm"

    # Get the sequences for train and val splits (in correct order)
    train_sequences = [sequences[i] for i in train_df.index]
    val_sequences = [sequences[i] for i in val_df.index]

    # Create datasets and loaders without shuffling
    train_pred_dataset = SequenceDataset(train_df, train_sequences, label_col=label_col)
    val_pred_dataset = SequenceDataset(val_df, val_sequences, label_col=label_col)

    train_pred_loader = DataLoader(
        train_pred_dataset, batch_size=config.batch_size * 4, collate_fn=collate_fn, shuffle=False
    )
    val_pred_loader = DataLoader(
        val_pred_dataset, batch_size=config.batch_size * 4, collate_fn=collate_fn, shuffle=False
    )

    # Get predictions for train and validation sets separately to preserve split info
    train_predictions, train_labels = trainer.get_predictions(train_pred_loader)
    val_predictions, val_labels = trainer.get_predictions(val_pred_loader)

    # Create results dataframe with all metadata columns using the actual split dataframes
    train_df_with_preds = train_df.copy()
    train_df_with_preds["predictions"] = train_predictions
    train_df_with_preds["NMDeff"] = train_labels
    train_df_with_preds["set"] = "train"

    val_df_with_preds = val_df.copy()
    val_df_with_preds["predictions"] = val_predictions
    val_df_with_preds["NMDeff"] = val_labels
    val_df_with_preds["set"] = "validation"

    # Combine train and validation results
    results_df = pd.concat([train_df_with_preds, val_df_with_preds], ignore_index=True)

    # Save predictions
    results_df.to_csv(trainer.output_dir / "predictions.csv", index=False)

    # Generate plots
    plot_predictions(results_df, trainer.output_dir / "predictions_plot.png")
    plot_LE_predictions(results_df, trainer.output_dir / "le_predictions_plot.png")

    wandb.finish()
    logger.success(f"DMS LE training complete. Model saved to {model_path}")


@app.command()
def train_dms_pe(
    pretrained_model_path: Optional[Path] = None,
    normalize: bool = True,
    max_eval_steps: int = 500,
):
    """Train model on DMS PE (50nt rule) dataset."""
    config = TrainerConfig()
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.normalize = normalize
    config.learning_rate = 0.0001
    config.pretrained_model_path = pretrained_model_path

    train_type = "DMS_PE"
    mode = "normalize" if normalize else ""
    run_name = _generate_run_name(f"{train_type}_{mode}", pretrained_model_path)

    # Setup trainer and data
    trainer = _setup_wandb_and_trainer(config, run_name, train_type, mode)
    data_path = PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv"
    sequences, metadata = setup_data(data_path, batch_size=config.batch_size, data_type="DMS")

    # Setup data loaders with stratification by gene
    train_loader, val_loader, train_df, val_df = _setup_dms_data_loaders(
        metadata, sequences, config, split_by="chr"
    )

    # Train and evaluate
    model_path = trainer.train_step_based(train_loader, val_loader)

    # Generate predictions and plots for PE dataset
    logger.info("Generating predictions for visualization...")

    # Create non-shuffled data loaders for predictions to preserve order
    label_col = "NMDeff" if not config.normalize else "NMDeff_Norm"

    # Get the sequences for train and val splits (in correct order)
    train_sequences = [sequences[i] for i in train_df.index]
    val_sequences = [sequences[i] for i in val_df.index]

    # Create datasets and loaders without shuffling
    train_pred_dataset = SequenceDataset(train_df, train_sequences, label_col=label_col)
    val_pred_dataset = SequenceDataset(val_df, val_sequences, label_col=label_col)

    train_pred_loader = DataLoader(
        train_pred_dataset, batch_size=config.batch_size * 4, collate_fn=collate_fn, shuffle=False
    )
    val_pred_loader = DataLoader(
        val_pred_dataset, batch_size=config.batch_size * 4, collate_fn=collate_fn, shuffle=False
    )

    # Get predictions for train and validation sets separately to preserve split info
    train_predictions, train_labels = trainer.get_predictions(train_pred_loader)
    val_predictions, val_labels = trainer.get_predictions(val_pred_loader)

    # Create results dataframe with all metadata columns using the actual split dataframes
    train_df_with_preds = train_df.copy()
    train_df_with_preds["predictions"] = train_predictions
    train_df_with_preds["NMDeff"] = train_labels
    train_df_with_preds["set"] = "train"

    val_df_with_preds = val_df.copy()
    val_df_with_preds["predictions"] = val_predictions
    val_df_with_preds["NMDeff"] = val_labels
    val_df_with_preds["set"] = "validation"

    # Combine train and validation results
    results_df = pd.concat([train_df_with_preds, val_df_with_preds], ignore_index=True)

    # Save predictions
    results_df.to_csv(trainer.output_dir / "predictions.csv", index=False)

    # Generate plots
    plot_predictions(results_df, trainer.output_dir / "predictions_plot.png")
    plot_PE_predictions(results_df, trainer.output_dir / "pe_predictions_plot.png")

    wandb.finish()
    logger.success(f"DMS PE training complete. Model saved to {model_path}")


@app.command()
def train_dms_combined(
    pretrained_model_path: Optional[Path] = None,
    normalize: bool = True,
    max_eval_steps: int = 1000,
):
    """Train model on combined DMS datasets."""
    config = TrainerConfig()
    config.max_steps = max_eval_steps * config.eval_every_steps
    config.normalize = normalize
    config.pretrained_model_path = pretrained_model_path

    train_type = "DMS_combined"
    mode = "normalize" if normalize else ""
    run_name = _generate_run_name(f"{train_type}_{mode}", pretrained_model_path)

    # Setup trainer and data
    trainer = _setup_wandb_and_trainer(config, run_name, train_type, mode)
    data_path = PROCESSED_DATA_DIR / "DMS_combined" / "fitness_combined.csv"
    sequences, metadata = setup_data(data_path, batch_size=config.batch_size, data_type="DMS")

    # Setup data loaders
    train_loader, val_loader, _, _ = _setup_dms_data_loaders(
        metadata, sequences, config, split_by="chr"
    )

    # Train and evaluate
    model_path = trainer.train_step_based(train_loader, val_loader)

    # Generate predictions and plots for combined dataset
    logger.info("Generating predictions for visualization...")

    # Get predictions for both train and validation sets
    train_predictions, train_labels = trainer.get_predictions(train_loader)
    val_predictions, val_labels = trainer.get_predictions(val_loader)

    # Create combined results dataframe with proper set labels
    train_results = pd.DataFrame(
        {"predictions": train_predictions, "NMDeff": train_labels, "set": "train"}
    )
    val_results = pd.DataFrame(
        {"predictions": val_predictions, "NMDeff": val_labels, "set": "validation"}
    )

    # Combine train and validation results
    results_df = pd.concat([train_results, val_results], ignore_index=True)

    # Save predictions
    results_df.to_csv(trainer.output_dir / "predictions.csv", index=False)

    # Generate plots
    plot_predictions(results_df, trainer.output_dir / "predictions_plot.png")

    wandb.finish()
    logger.success(f"DMS combined training complete. Model saved to {model_path}")


# Helper functions
def _setup_dms_data_loaders(
    metadata: pd.DataFrame,
    sequences: list,
    config: TrainerConfig,
    stratify_by: Optional[str] = None,
    split_by: Optional[str] = None,
) -> tuple:
    """Setup train/validation data loaders for DMS datasets."""
    label_col = "NMDeff" if not config.normalize else "NMDeff_Norm"

    # Split data based on chromosome if split_by="chr" is specified
    if split_by == "chr":
        if "chr" not in metadata.columns:
            raise ValueError("Cannot split by chromosome: 'chr' column not found in metadata")
        
        # Split based on whether chromosome is in VAL_CHRS
        val_mask = metadata["chr"].isin(VAL_CHRS)
        train_df = metadata[~val_mask]
        val_df = metadata[val_mask]
        
        train_sequences = [sequences[i] for i in train_df.index]
        val_sequences = [sequences[i] for i in val_df.index]
        
        logger.info(f"Splitting by chromosome: validation chromosomes are {VAL_CHRS}")
    else:
        # Use standard train_test_split with optional stratification
        stratify = metadata[stratify_by] if stratify_by and stratify_by in metadata.columns else None
        train_df, val_df, train_sequences, val_sequences = train_test_split(
            metadata, sequences, test_size=0.2, random_state=SEED, stratify=stratify
        )
    
    logger.info(f"Training on {len(train_df)} sequences, validating on {len(val_df)} sequences")
    logger.info(f"Using label column: {label_col}")

    # Create datasets
    train_dataset = SequenceDataset(train_df, train_sequences, label_col=label_col)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataset = SequenceDataset(val_df, val_sequences, label_col=label_col)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    return train_loader, val_loader, train_df, val_df


def _setup_cv_fold_data(split: dict, sequences: list, config: TrainerConfig) -> tuple:
    """Setup data loaders for cross-validation fold."""
    label_col = "NMDeff" if not config.normalize else "NMDeff_Norm"

    # Extract fold data
    train_df, test_df = split["train_df"], split["test_df"]
    train_sequences = [sequences[i] for i in train_df.index]
    test_sequences = [sequences[i] for i in test_df.index]

    # Create datasets and loaders
    train_dataset = SequenceDataset(train_df, train_sequences, label_col=label_col)
    test_dataset = SequenceDataset(test_df, test_sequences, label_col=label_col)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    return train_loader, test_loader


def _add_chromosome_column(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add chromosome information to metadata DataFrame based on gene names."""
    genome = Genome(GENCODE_VERSION)
    
    # Create gene to chromosome mapping
    gene_to_chr = {}
    unique_genes = metadata["gene"].unique()
    
    for gene_symbol in unique_genes:
        matching_genes = [g for g in genome.genes if g.name == gene_symbol]
        if matching_genes:
            gene_to_chr[gene_symbol] = matching_genes[0].chromosome
        else:
            logger.warning(f"Gene {gene_symbol} not found in genome, assigning 'unknown'")
            gene_to_chr[gene_symbol] = "unknown"
    
    # Add chromosome column to metadata
    metadata = metadata.copy()
    metadata["chr"] = metadata["gene"].map(gene_to_chr)
    
    logger.info(f"Chromosome distribution: {metadata['chr'].value_counts().to_dict()}")
    
    return metadata


if __name__ == "__main__":
    app()
