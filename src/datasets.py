import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        # Apply baseline correction
        X = self.apply_baseline_correction(X)

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @staticmethod
    def apply_baseline_correction(X):
        # Assume X is a 2D tensor with shape (num_channels, seq_len)
        num_channels, seq_len = X.shape
        corrected_X = torch.empty_like(X)
        
        for channel in range(num_channels):
            segment = X[channel, :]
            # Calculate the baseline value as the mean of the smallest 10% of the segment
            baseline_value = segment.topk(int(len(segment) * 0.1), largest=False).values.mean()
            # Subtract the baseline value from the segment
            corrected_X[channel, :] = segment - baseline_value

        return corrected_X

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
