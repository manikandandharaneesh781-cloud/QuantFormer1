from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COLUMN_INDEX = FEATURE_COLUMNS.index("Close")


class StockSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


def prepare_datasets(
    data: pd.DataFrame,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Create chronological sequence datasets.
    The scaler is fitted only on the training portion to reduce leakage.
    Each sample uses `sequence_length` rows to predict the following Close.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    values = data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    row_count = len(values)

    train_end = int(row_count * train_ratio)
    validation_end = int(row_count * (train_ratio + validation_ratio))

    if train_end <= sequence_length or validation_end <= train_end:
        raise ValueError("Not enough rows for the selected sequence length and split ratios.")

    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values).astype(np.float32)

    x_train, y_train = [], []
    x_validation, y_validation = [], []
    x_test, y_test, test_dates = [], [], []

    # target_index is the row being predicted; input ends one row before it.
    for target_index in range(sequence_length, row_count):
        x = scaled_values[target_index - sequence_length:target_index]
        y = scaled_values[target_index, TARGET_COLUMN_INDEX]

        if target_index < train_end:
            x_train.append(x)
            y_train.append(y)
        elif target_index < validation_end:
            x_validation.append(x)
            y_validation.append(y)
        else:
            x_test.append(x)
            y_test.append(y)
            test_dates.append(data.index[target_index])

    def as_arrays(xs, ys):
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    train_x, train_y = as_arrays(x_train, y_train)
    validation_x, validation_y = as_arrays(x_validation, y_validation)
    test_x, test_y = as_arrays(x_test, y_test)

    if min(len(train_x), len(validation_x), len(test_x)) == 0:
        raise ValueError("One of the data splits is empty. Download more history.")

    return {
        "train": StockSequenceDataset(train_x, train_y),
        "validation": StockSequenceDataset(validation_x, validation_y),
        "test": StockSequenceDataset(test_x, test_y),
        "scaler": scaler,
        "test_dates": pd.DatetimeIndex(test_dates),
        "test_x": test_x,
        "test_y": test_y,
        "train_end": train_end,
        "validation_end": validation_end,
    }


def inverse_close_scale(scaled_values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Convert normalized Close values back to the original price scale."""
    scaled_values = np.asarray(scaled_values).reshape(-1)
    close_min = scaler.data_min_[TARGET_COLUMN_INDEX]
    close_range = scaler.data_range_[TARGET_COLUMN_INDEX]
    return scaled_values * close_range + close_min


def scale_latest_window(data: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int):
    values = data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if len(values) < sequence_length:
        raise ValueError(f"At least {sequence_length} rows are required.")
    scaled = scaler.transform(values).astype(np.float32)
    return scaled[-sequence_length:][None, :, :]
