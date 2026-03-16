import random
import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from pathlib import Path
import shutil


def collate_fn(batch):
    # Unzip the batch into separate lists
    sequences, lengths, labels = zip(*batch)

    # Find max length in this batch
    # Sequences are now in (length, channels) format
    max_len = max(seq.shape[0] for seq in sequences)

    # Pad sequences to max length
    padded_sequences = []
    for seq in sequences:
        # Get current sequence dimensions: (seq_len, n_channels)
        seq_len, n_channels = seq.shape
        # Create padding: (max_len - seq_len, n_channels)
        padding = torch.zeros((max_len - seq_len, n_channels))
        # Concatenate original sequence with padding along length dimension
        padded_seq = torch.cat([seq, padding], dim=0)
        padded_sequences.append(padded_seq)

    # Stack all sequences into a single tensor: (batch_size, max_len, n_channels)
    sequences_tensor = torch.stack(padded_sequences)
    lengths_tensor = torch.stack(lengths)
    labels_tensor = torch.stack(labels)

    return sequences_tensor, lengths_tensor, labels_tensor


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


def load_model(model, path, optimizer=None, device="cuda"):
    """Load model checkpoint

    Args:
        model: NMDOrthrus model instance
        path: path to checkpoint file
        optimizer: optional optimizer instance
        device: device to load model to

    Returns:
        epoch: epoch number of loaded checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" not in checkpoint:
        checkpoint["model_state_dict"] = checkpoint.get("state_dict", {})

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    #logger.info(f"Model loaded from {path}")
    return checkpoint.get("epoch", 0)


def relative_squared_error(y_true, y_pred):
    """Calculate Relative Squared Error (RSE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # Avoid division by zero
    if denominator == 0:
        return float("inf") if numerator > 0 else 0.0

    return numerator / denominator


def loess_smooth(x, y, frac=0.3, degree=2):
    """
    Apply LOESS-like smoothing to data using local polynomial regression.

    Args:
        x: array of x values
        y: array of y values
        frac: fraction of data to use for local regression (window size)
        degree: degree of polynomial for local fit

    Returns:
        x_smooth: sorted x values
        y_smooth: smoothed y values
    """
    # Sort data by x values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Calculate window size
    n = len(x)
    window_size = max(int(n * frac), 10)

    x_smooth = []
    y_smooth = []

    for i in range(n):
        # Define window around current point
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n, i + window_size // 2)

        # Get local data
        x_local = x_sorted[start_idx:end_idx]
        y_local = y_sorted[start_idx:end_idx]

        # Fit local polynomial
        if len(x_local) > degree:
            poly_features = PolynomialFeatures(degree=degree)
            poly_reg = Pipeline([("poly", poly_features), ("linear", LinearRegression())])
            poly_reg.fit(x_local.reshape(-1, 1), y_local)
            y_pred = poly_reg.predict([[x_sorted[i]]])[0]
        else:
            y_pred = np.mean(y_local)

        x_smooth.append(x_sorted[i])
        y_smooth.append(y_pred)

    return np.array(x_smooth), np.array(y_smooth)


def prepare_for_interp(x, y):
    """
    Remove duplicates and sort arrays for interpolation.

    Args:
        x: array of x values
        y: array of y values

    Returns:
        unique_x: unique sorted x values
        unique_y: corresponding y values
    """
    # Combine and sort by x
    combined = np.column_stack([x, y])
    combined = combined[np.argsort(combined[:, 0])]

    # Remove duplicates (keep first occurrence)
    unique_x, unique_indices = np.unique(combined[:, 0], return_index=True)
    unique_y = combined[unique_indices, 1]

    return unique_x, unique_y


def copy_preprocessing_configs(model_output_dir: Path, processed_data_dir: Path):
    """
    Copy all preprocessing configuration files to model output directory.
    
    Args:
        model_output_dir: Directory where the model is saved
        processed_data_dir: Directory containing processed PTC datasets
    """
    datasets = ["somatic_TCGA", "germline_TCGA", "GTEx"]
    
    for dataset_name in datasets:
        preprocessing_config_path = processed_data_dir / "PTC" / f"{dataset_name}_preprocessing_config.txt"
        
        if preprocessing_config_path.exists():
            dest_path = model_output_dir / f"preprocessing_parameters_{dataset_name}.txt"
            shutil.copy(preprocessing_config_path, dest_path)
            logger.info(f"Copied preprocessing configuration for {dataset_name} to {dest_path}")
        else:
            logger.warning(f"Preprocessing configuration not found: {preprocessing_config_path}")
