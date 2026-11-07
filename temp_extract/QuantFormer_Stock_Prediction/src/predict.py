from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data_loader import download_stock_data
from src.model import QuantFormer
from src.preprocessing import scale_latest_window, inverse_close_scale

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Predict the next available trading-session close.")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    checkpoint_path = ROOT / "models" / f"quantformer_{ticker}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}. Train it first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data = download_stock_data(ticker, args.period)

    latest_window = scale_latest_window(
        data,
        checkpoint["scaler"],
        checkpoint["sequence_length"],
    )
    input_tensor = torch.tensor(latest_window, dtype=torch.float32, device=device)

    model = QuantFormer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        scaled_prediction = model(input_tensor).cpu().numpy().reshape(-1)

    predicted_close = inverse_close_scale(scaled_prediction, checkpoint["scaler"])[0]
    latest_close = float(data["Close"].iloc[-1])
    latest_date = data.index[-1].date()

    print("\\n========== NEXT SESSION ESTIMATE ==========")
    print(f"Ticker: {ticker}")
    print(f"Latest available trading date: {latest_date}")
    print(f"Latest adjusted close: {latest_close:.2f}")
    print(f"Estimated next-session close: {predicted_close:.2f}")
    print("\\nThis is a model estimate, not a guaranteed price or investment advice.")


if __name__ == "__main__":
    main()
