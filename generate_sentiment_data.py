"""
generate_sentiment_data.py

Generate synthetic sentiment signals for testing the Risk & Valuation Agent.
"""

import pandas as pd
import numpy as np

def generate_synthetic_sentiment_signals(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "sentiment_signals.csv",
):
    """
    Generate synthetic sentiment signals aligned with trading days.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    
    np.random.seed(42)
    
    # Generate base sentiment with regime changes
    n = len(date_range)
    trend = np.linspace(-0.1, 0.1, n)
    cycles = 0.15 * np.sin(2 * np.pi * np.arange(n) / 252)
    noise = np.random.normal(0, 0.05, n)
    
    S_raw = trend + cycles + noise
    
    # Rolling average and volatility
    df = pd.DataFrame({
        "S_raw": S_raw,
    }, index=date_range)
    
    df["S_MA"] = df["S_raw"].rolling(window=5, min_periods=1).mean()
    df["S_Vol"] = df["S_raw"].rolling(window=5, min_periods=2).std()
    
    df.index.name = "trading_date"
    df.to_csv(output_path)
    
    print(f"✓ Synthetic sentiment signals saved to {output_path}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Observations: {len(df)}")
    print(f"  S_raw range: [{df['S_raw'].min():.3f}, {df['S_raw'].max():.3f}]")

if __name__ == "__main__":
    generate_synthetic_sentiment_signals()