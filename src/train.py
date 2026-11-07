from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_loader import download_stock_data
from src.model import QuantFormer
from src.preprocessing import prepare_datasets, inverse_close_scale

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    nonzero = np.abs(actual) > 1e-8
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_model(model, train_loader, validation_loader, device, epochs, learning_rate, patience):
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_losses.append(loss.item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                validation_losses.append(criterion(model(batch_x), batch_y).item())

        train_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the QuantFormer-inspired stock forecasting model.")
    parser.add_argument("--ticker", default="RELIANCE.NS", help="Yahoo Finance ticker, e.g. RELIANCE.NS or AAPL")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 2y, 5y, max")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    set_seed()
    ticker = args.ticker.strip().upper()
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Downloading historical data for {ticker}...")
    data = download_stock_data(ticker, args.period)
    print(f"Downloaded {len(data)} daily records: {data.index.min().date()} to {data.index.max().date()}")

    prepared = prepare_datasets(data, sequence_length=args.sequence_length)
    train_loader = DataLoader(prepared["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(prepared["validation"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(prepared["test"], batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = QuantFormer().to(device)
    model = train_model(
        model, train_loader, validation_loader, device,
        args.epochs, args.learning_rate, args.patience
    )

    model.eval()
    scaled_predictions, scaled_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x.to(device)).cpu().numpy().reshape(-1)
            scaled_predictions.extend(predictions)
            scaled_actuals.extend(batch_y.numpy().reshape(-1))

    scaler = prepared["scaler"]
    actual_prices = inverse_close_scale(np.array(scaled_actuals), scaler)
    predicted_prices = inverse_close_scale(np.array(scaled_predictions), scaler)
    metrics = calculate_metrics(actual_prices, predicted_prices)

    print("\\n========== TEST PERFORMANCE ==========")
    for metric_name, metric_value in metrics.items():
        suffix = "%" if metric_name == "MAPE" else ""
        print(f"{metric_name}: {metric_value:.4f}{suffix}")

    checkpoint_path = MODEL_DIR / f"quantformer_{ticker}.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "ticker": ticker,
        "period": args.period,
        "sequence_length": args.sequence_length,
        "model_config": {
            "input_features": 5,
            "model_dimension": 64,
            "attention_heads": 4,
            "encoder_layers": 2,
            "feedforward_dimension": 128,
            "dropout": 0.1,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    results = __import__("pandas").DataFrame({
        "Date": prepared["test_dates"],
        "Actual_Close": actual_prices,
        "Predicted_Close": predicted_prices,
    })
    csv_path = OUTPUT_DIR / f"{ticker}_test_predictions.csv"
    results.to_csv(csv_path, index=False)

    plot_path = OUTPUT_DIR / f"{ticker}_predictions.png"
    plt.figure(figsize=(12, 6))
    plt.plot(results["Date"], results["Actual_Close"], label="Actual Close")
    plt.plot(results["Date"], results["Predicted_Close"], label="Predicted Close")
    plt.title(f"{ticker} - QuantFormer Test Predictions")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closing price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\\nSaved model: {checkpoint_path}")
    print(f"Saved test predictions: {csv_path}")
    print(f"Saved prediction chart: {plot_path}")


if __name__ == "__main__":
    main()
