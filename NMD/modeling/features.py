import pickle
import numpy as np
import pandas as pd


from pathlib import Path
from torch.utils.data import DataLoader
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.config import VAL_CHRS
from NMD.utils import collate_fn
from loguru import logger
import typer

app = typer.Typer()


def setup_data(
    data_path: Path,
    batch_size: int,
    data_type: str = "PTC",
    normalize: bool = False,
):
    """
    Setup datasets and dataloaders for training and evaluation.

    Args:
        data_path (Path): Path to the data (without extension).
        batch_size (int): Batch size for DataLoader.
        data_type (str): 'PTC' or 'DMS'.
        normalize (bool): Whether to use normalized labels.

    Returns:
        Tuple of DataLoaders and DataFrames, depending on data_type and holdout_test.
    """
    if data_type == "PTC":
        sequences, metadata = load_data(data_path)
        train_df, val_df, train_sequences, val_sequences = split_data(
            metadata, sequences
        )

        col = "NMDeff_Norm" if normalize else "NMDeff" 
        train_dataset = SequenceDataset(train_df, train_sequences, label_col=col)
        val_dataset = SequenceDataset(val_df, val_sequences, label_col=col)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,  # Ensure complete batches for gradient accumulation
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 4, collate_fn=collate_fn)

        return train_loader, val_loader, train_df, val_df

    elif data_type == "DMS":
        with open(f"{data_path.parent}/processed_sequences.pkl", "rb") as f:
            sequences = pickle.load(f)
        logger.info(f"Successfully loaded {len(sequences)} sequences")
        metadata = pd.read_csv(data_path.parent / "fitness.csv")
        return sequences, metadata
    elif data_type == "DMS_minigene":
        with open(f"{data_path.parent}/processed_sequences_minigene.npy", "rb") as f:
            sequences = np.load(f, allow_pickle=True)
        logger.info(f"Successfully loaded {len(sequences)} minigene sequences")
        metadata = pd.read_csv(data_path.parent / "fitness_minigene.csv")
        return sequences, metadata
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def load_data(data_path):
    """Load the sequence data and associated metadata"""
    # Load pickled sequences
    with open(f"{data_path}.pkl", "rb") as f:
        sequences = pickle.load(f)

    # Load metadata
    metadata = pd.read_csv(f"{data_path}.csv")

    return sequences, metadata


def split_data(df, sequences):
    """Split data into train, validation and optionally test sets based on chromosomes."""

    chr_col = "chr" if "chr" in df.columns else "chromosome"

    # Filter training data to exclude validation and test chromosomes
    train_df = df[~df[chr_col].isin(VAL_CHRS)]
    train_sequences = [sequences[i] for i in train_df.index]
    logger.info(f"Train: {len(train_df)} samples")

    # Filter validation data based on specified chromosomes
    val_df = df[df[chr_col].isin(VAL_CHRS)]
    val_sequences = [sequences[i] for i in val_df.index]
    logger.info(f"Val: {len(val_df)} samples")

    return train_df, val_df, train_sequences, val_sequences


def split_genes_data(df, sequences, train_ratio=0.7, val_ratio=0.15):
    """Split data based on genes rather than random samples"""
    df = df.copy()
    df = df[df["wild_type"] == 0]  # Filter non-wild type samples

    unique_genes = df["gene"].unique()
    np.random.shuffle(unique_genes)

    n_genes = len(unique_genes)
    train_idx = int(n_genes * train_ratio)
    val_idx = int(n_genes * (train_ratio + val_ratio))

    train_genes = unique_genes[:train_idx]
    val_genes = unique_genes[train_idx:val_idx]
    test_genes = unique_genes[val_idx:]

    train_df = df[df["gene"].isin(train_genes)]
    val_df = df[df["gene"].isin(val_genes)]
    test_df = df[df["gene"].isin(test_genes)]

    train_sequences = sequences[train_df.index]
    val_sequences = sequences[val_df.index]
    test_sequences = sequences[test_df.index]

    return (train_df, val_df, test_df), (train_sequences, val_sequences, test_sequences)


def get_gene_cv_splits(df, n_splits=10):
    """Generate cross-validation splits based on genes"""
    unique_genes = df["gene"].unique()
    np.random.shuffle(unique_genes)

    # Split genes into n_splits approximately equal groups
    gene_splits = np.array_split(unique_genes, n_splits)

    splits = []
    for i in range(n_splits):
        # Use split i as test set
        test_genes = gene_splits[i]
        # Use all other genes for training (no validation set)
        train_genes = np.concatenate([gene_splits[j] for j in range(n_splits) if j != i])

        splits.append({"train_genes": train_genes, "test_genes": test_genes})

    return splits


def split_by_genes(df, sequences, train_genes, val_genes=None, test_genes=None):
    """Split data based on provided gene lists"""
    train_df = df[df["gene"].isin(train_genes)]
    train_sequences = [sequences[i] for i in train_df.index]
    logger.info(f"Train: {len(train_df)} samples, {len(train_genes)} genes")

    if val_genes is not None:
        val_df = df[df["gene"].isin(val_genes)]
        val_sequences = [sequences[i] for i in val_df.index]
        logger.info(f"Val: {len(val_df)} samples, {len(val_genes)} genes")
    else:
        val_df = None
        val_sequences = None

    if test_genes is not None:
        test_df = df[df["gene"].isin(test_genes)]
        test_sequences = [sequences[i] for i in test_df.index]
        logger.info(f"Test: {len(test_df)} samples, {len(test_genes)} genes")

        if val_genes is not None:
            return (train_df, val_df, test_df), (train_sequences, val_sequences, test_sequences)
        else:
            return (train_df, test_df), (train_sequences, test_sequences)

    return (train_df, val_df), (train_sequences, val_sequences)


if __name__ == "__main__":
    app()
