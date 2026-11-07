# QuantFormer: Stock Market Price Prediction

A beginner-friendly, QuantFormer-inspired Transformer model for next-trading-day stock closing-price forecasting.

> This repository implements a Transformer time-series forecasting model inspired by the QuantFormer project idea. It does not claim to reproduce a particular published QuantFormer paper or proprietary architecture.

## Features
- Downloads historical daily OHLCV data from Yahoo Finance using `yfinance`.
- Uses chronological train/validation/test splits.
- Fits feature scaling on training data only to reduce data leakage.
- Trains a Transformer Encoder model in PyTorch.
- Reports MAE, RMSE and MAPE on the held-out test set.
- Saves the trained model and prediction plot.
- Supports US and Indian Yahoo Finance ticker symbols.

## Setup (Windows PowerShell)

```powershell
cd QuantFormer1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands without activating the environment by using `.venv\Scripts\python.exe` instead of `python`.

## Train

Example for Reliance Industries (NSE):

```powershell
python -m src.train --ticker RELIANCE.NS --period 5y
```

Example for Apple:

```powershell
python -m src.train --ticker AAPL --period 5y
```

The script creates:
- `models/quantformer_RELIANCE.NS.pth`
- `outputs/RELIANCE.NS_predictions.png`
- `outputs/RELIANCE.NS_test_predictions.csv`

## Evaluate an existing model

```powershell
python -m src.evaluate --ticker RELIANCE.NS --period 5y
```

Use the same ticker and period used for training.

## Predict the next trading day

```powershell
python -m src.predict --ticker RELIANCE.NS --period 5y
```

The output is a model estimate, not a guaranteed market price or investment recommendation. Markets are affected by news, liquidity, macroeconomic conditions and many other factors. Yahoo Finance data availability and timing may vary.

## Project structure

```text
QuantFormer1/
├── dataset/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important
- Use a valid Yahoo Finance ticker. Indian examples: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `^NSEI`.
- Training requires an internet connection to download data.
- A one-step forecast is an estimate for the next available trading session, not necessarily the next calendar day.
