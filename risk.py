"""
Valuation & Risk Agent
Transforms macro + sentiment signals into expected return,
risk metrics (volatility, VaR), and allocation regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class AssetConfig:
    name: str
    var_level: float = 0.95


class ValuationRiskAgent:
    """
    Valuation & Risk Agent.

    Inputs:
      - macro_signals.csv   (from MacroAgent)
      - sentiment_signals.csv (from NewsSentimentAgent)
      - market returns (ETF or bond index)

    Outputs:
      - expected return μ_t
      - volatility σ_t
      - VaR
      - regime classification
      - confidence score
    """

    def __init__(
        self,
        macro_weight: float = 0.6,
        sentiment_weight: float = 0.4,
        reg_window: int = 252 * 3,
        vol_window: int = 63,
        var_z: float = 1.65,
    ):
        self.macro_weight = macro_weight
        self.sentiment_weight = sentiment_weight
        self.reg_window = reg_window
        self.vol_window = vol_window
        self.var_z = var_z

    def run(
        self,
        macro_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        returns: pd.Series,
        output_path: Optional[str] = "valuation_risk_signals.csv",
    ) -> pd.DataFrame:
        data = self._align_inputs(macro_df, sentiment_df, returns)
        data = self._compute_fused_signal(data)
        data = self._estimate_expected_return(data)
        data = self._estimate_risk(data)
        data = self._classify_regime(data)
        data = self._compute_confidence(data)

        if output_path:
            data.to_csv(output_path, index=True)
            print(f"Valuation & Risk signals saved to {output_path}")

        return data

    def _align_inputs(
        self,
        macro: pd.DataFrame,
        sentiment: pd.DataFrame,
        returns: pd.Series,
    ) -> pd.DataFrame:
        df = pd.concat(
            [
                macro[["M_score"]],
                sentiment[["S_MA"]],
                returns.rename("ret"),
            ],
            axis=1,
        ).dropna()
        return df

    def _compute_fused_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = (
            self.macro_weight * df["M_score"]
            + self.sentiment_weight * df["S_MA"]
        )
        return df

    def _estimate_expected_return(self, df: pd.DataFrame) -> pd.DataFrame:
        mu = []

        for i in range(len(df)):
            if i < self.reg_window:
                mu.append(np.nan)
                continue

            window = df.iloc[i - self.reg_window : i]
            X = np.column_stack(
                [np.ones(len(window)), window["signal"].values]
            )
            y = window["ret"].shift(-1).dropna().values

            if len(y) < 20:
                mu.append(np.nan)
                continue

            beta = np.linalg.lstsq(X[:-1], y, rcond=None)[0]
            mu.append(beta[0] + beta[1] * df["signal"].iloc[i])

        df["mu"] = mu
        return df

    def _estimate_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sigma"] = df["ret"].rolling(self.vol_window).std()
        df["VaR"] = df["mu"] - self.var_z * df["sigma"]
        return df

    def _classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        conditions = []
        regimes = []

        for _, row in df.iterrows():
            if pd.isna(row["mu"]) or pd.isna(row["sigma"]):
                regimes.append("Neutral")
            elif row["sigma"] > df["sigma"].quantile(0.8):
                regimes.append("Unstable")
            elif row["mu"] > 0 and row["VaR"] > 0:
                regimes.append("Risk-On")
            elif row["mu"] < 0 and row["VaR"] < 0:
                regimes.append("Risk-Off")
            else:
                regimes.append("Neutral")

        df["regime"] = regimes
        return df

    def _compute_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        k = 0.1
        sigma_star = df["sigma"].median()

        df["confidence"] = (
            df["signal"].abs() / (df["signal"].abs() + k)
        ) * np.exp(-df["sigma"] / sigma_star)

        return df
