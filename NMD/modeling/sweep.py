import wandb
import typer
from loguru import logger
import time

import matplotlib

matplotlib.use("Agg")

from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.Trainer import Trainer
from NMD.modeling.evaluation import ModelEvaluator
from NMD.modeling.features import setup_data
from NMD.config import PROCESSED_DATA_DIR, WANDB_PROJECT
from NMD.utils import copy_preprocessing_configs

app = typer.Typer()


# Sweep configurations for different datasets
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "Best Val Loss"},
    "parameters": {
        "learning_rate": {"values": [0.00001, 0.00005, 0.0001, 0.001]},
        "weight_decay": {"values": [1e-5, 1e-3, 0]},
        "orthrus_weight_decay": {"values": [0, 1e-3, 1e-5]},
        "lr_gamma": {"values": [0.9, 0.95, 0.99]},
        "accumulation_steps": {"values": [8, 64, 256]},
        "dnn_hidden_dims": {
            "values": [
                [256, 64],
                [512, 256, 128, 64],
                [1024, 512, 256, 128, 64],
            ]
        },
        "dnn_dropout": {"values": [0.0, 0.1, 0.2]},
        "max_norm": {"values": [1.0, 2.0, 3.0]},
        "warmup_steps": {"values": [50, 100, 10000]},
        "Orthrus_MLM": {"values": [True, False]},
        "activation_function": {"values": ["relu", "gelu"]},
        "use_layer_norm": {"values": [True, False]},
        "loss_type": {"values": ["mse", "huber"]},
        "huber_delta": {"values": [0.5, 1.0, 2.0]},
    },
}


def _update_config_from_wandb(config: TrainerConfig) -> TrainerConfig:
    """Update TrainerConfig with wandb sweep parameters."""
    if not wandb.run:
        return config

    # Get wandb config
    wandb_config = wandb.config

    # Update config with sweep parameters
    for param_name, param_value in wandb_config.items():
        if hasattr(config, param_name):
            setattr(config, param_name, param_value)
            logger.info(f"Updated {param_name} = {param_value}")

    return config


def train_ptc_sweep():
    """Training function for PTC dataset sweep."""
    # Initialize wandb run for this sweep iteration
    with wandb.init() as run:
        config = TrainerConfig()
        config = _update_config_from_wandb(config)

        logger.info("Starting PTC training with sweep parameters")

        # Set up data
        data_path = PROCESSED_DATA_DIR / "PTC/somatic_TCGA"
        train_loader, val_loader, _, _ = setup_data(
            data_path, config.batch_size, "PTC", normalize=True
        )

        # Create trainer
        trainer = Trainer(config, "PTC", "Sweep")

        # Train model
        try:
            model_path = trainer.train_step_based(train_loader, val_loader)
        except:
            logger.exception("Training failed during sweep run")
            time.sleep(5)  # If we launch too early, GPU resources might not be ready
            raise

        # Copy all preprocessing configurations to model output directory
        copy_preprocessing_configs(model_path.parent, PROCESSED_DATA_DIR)

        # Comprehensive evaluation
        evaluator = ModelEvaluator(model_path, config)
        evaluator.evaluate_all_ptc_datasets()
        
        # Evaluate on DMS datasets
        logger.info("Evaluating on DMS datasets")
        evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv", dataset_name="PE")
        evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv", dataset_name="LE")
        evaluator.evaluate_dms_dataset(PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv", dataset_name="SP")

        # Log final model path
        wandb.log({"model_path": str(model_path)})
        logger.success(f"PTC sweep run complete. Model saved to {model_path}")


@app.command()
def run_sweep(count: int = typer.Option(50, help="Number of runs in the sweep")):
    """Create and run a wandb sweep."""
    logger.info("Initializing wandb sweep...")

    # Create the sweep
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    logger.success(f"Created sweep with ID: {sweep_id}")
    logger.info(f"Starting sweep with {count} runs")

    # Run the sweep
    wandb.agent(sweep_id, function=train_ptc_sweep, count=count, project=WANDB_PROJECT)
    logger.success(f"Sweep completed!")


if __name__ == "__main__":
    app()
