import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, df, sequences, label_col="NMDeff"):
        self.df = df
        self.sequences = sequences
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence = torch.tensor(sequence, dtype=torch.float)
        
        # All data is saved in (L, 6) format for Orthrus channel_last=True
        length = torch.tensor(sequence.shape[0], dtype=torch.long)
        label = torch.tensor(self.df.iloc[idx][self.label_col], dtype=torch.float)
        return sequence, length, label
