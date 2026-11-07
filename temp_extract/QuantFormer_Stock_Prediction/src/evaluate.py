from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import download_stock_data
from src.model import QuantFormer
from src.preprocessing import prepare_datasets, inverse_close_scale
from src.train import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved QuantFormer model.")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    checkpoint_path = ROOT / "models" / f"quantformer_{ticker}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}. Train it first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if checkpoint["period"] != args.period:
        print("Warning: evaluation period differs from the period used during training.")

    data = download_stock_data(ticker, args.period)
    prepared = prepare_datasets(
        data,
        sequence_length=checkpoint["sequence_length"],
    )

    model = QuantFormer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(prepared["test"], batch_size=args.batch_size, shuffle=False)
    predictions, actuals = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions.extend(model(batch_x.to(device)).cpu().numpy().reshape(-1))
            actuals.extend(batch_y.numpy().reshape(-1))

    actual_prices = inverse_close_scale(np.array(actuals), checkpoint["scaler"])
    predicted_prices = inverse_close_scale(np.array(predictions), checkpoint["scaler"])

    print("\\n========== TEST PERFORMANCE ==========")
    for name, value in calculate_metrics(actual_prices, predicted_prices).items():
        print(f"{name}: {value:.4f}{'%' if name == 'MAPE' else ''}")


if __name__ == "__main__":
    main()
