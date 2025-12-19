"""
risk_valuation_agent.py

Valuation & Risk Agent for fixed-income portfolio allocation.
Integrates macro and sentiment signals to generate expected returns,
volatility estimates, and regime-based allocation recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats




@dataclass
class RiskMetrics:
    """Container for risk metrics at time t."""
    expected_return: float
    volatility: float
    var_95: float
    confidence: float
    regime: str


class RiskValuationAgent:
    """
    Valuation & Risk Agent.
    
    Integrates:
      - Macro signals (M_t) from Macro & Fundamentals Agent
      - Sentiment signals (S_t) from News & Sentiment Agent
      - ETF price data for return calculation
    
    Outputs:
      - Expected return (μ_t)
      - Volatility (σ_t)
      - Value-at-Risk (VaR_95)
      - Market regime classification
      - Portfolio allocation recommendations
    """

    REGIME_THRESHOLDS = {
        "mu_positive": 0.0,
        "var_threshold": -0.02,
        "high_vol": 0.35,
        "high_confidence": 0.75,
    }

    def __init__(
        self,
        w_macro: float = 0.6,
        w_sentiment: float = 0.4,
        return_window: int = 60,  # ~3 months daily
        vol_window: int = 20,     # ~1 month daily
        ols_window: int = 756,    # ~3 years daily
        k_confidence: float = 0.5,
        sigma_star: float = 0.20,
    ):
        """
        Parameters
        ----------
        w_macro : float
            Weight for macro signal in fusion (w1).
        w_sentiment : float
            Weight for sentiment signal in fusion (w2).
        return_window : int
            Rolling window for return volatility estimation.
        vol_window : int
            Rolling window for short-term volatility.
        ols_window : int
            Window for rolling OLS regression (μ_t = α + β·s_t).
        k_confidence : float
            Smoothness parameter for confidence score.
        sigma_star : float
            Volatility scaling parameter for confidence.
        """
        if not np.isclose(w_macro + w_sentiment, 1.0):
            raise ValueError("w_macro + w_sentiment must equal 1.0")
        
        self.w_macro = w_macro
        self.w_sentiment = w_sentiment
        self.return_window = return_window
        self.vol_window = vol_window
        self.ols_window = ols_window
        self.k_confidence = k_confidence
        self.sigma_star = sigma_star

    def run(
        self,
        macro_signals: pd.DataFrame,
        sentiment_signals: pd.DataFrame,
        etf_tickers: list[str] = ["LQD", "HYG", "IEF"],
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        output_path: Optional[str] = "risk_valuation_signals.csv",
    ) -> pd.DataFrame:
        """
        Execute the Risk & Valuation Agent pipeline.
        
        Parameters
        ----------
        macro_signals : pd.DataFrame
            Output from Macro Agent, must contain 'M_score'.
        sentiment_signals : pd.DataFrame
            Output from Sentiment Agent, must contain 'S_raw' or 'S_MA'.
        etf_tickers : list[str]
            List of ETF tickers to analyze.
        start_date : str
        end_date : str
        output_path : Optional[str]
            Path to save output CSV.
        
        Returns
        -------
        pd.DataFrame
            Risk & valuation signals with columns:
            - date, ticker, macro_signal, sentiment_signal, fused_signal
            - expected_return, volatility, var_95, confidence, regime
            - action_recommendation
        """
        print("=" * 60)
        print("Risk & Valuation Agent - Starting Pipeline")
        print("=" * 60)

        # Step 1: Fetch ETF price data
        price_data = self._fetch_etf_data(etf_tickers, start_date, end_date)
        
        # Step 2: Align signals
        aligned = self._align_signals(macro_signals, sentiment_signals, price_data)
        
        # Step 3: Compute fused signal
        aligned = self._compute_fused_signal(aligned)
        
        # Step 4: Estimate expected returns
        aligned = self._estimate_expected_returns(aligned)
        
        # Step 5: Estimate risk metrics
        aligned = self._estimate_risk_metrics(aligned)
        
        # Step 6: Classify regimes
        aligned = self._classify_regimes(aligned)
        
        # Step 7: Generate recommendations
        aligned = self._generate_recommendations(aligned)

        if output_path:
            aligned.to_csv(output_path, index=False)
            print(f"\n✓ Risk & valuation signals saved to {output_path}")

        self._print_summary(aligned)
        
        return aligned

    def _fetch_etf_data(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily adjusted close prices for ETFs."""
        print(f"\nFetching ETF data for {tickers}...")
        
        all_data = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if df.empty:
                    warnings.warn(f"No data for {ticker}")
                    continue
                
                df = df[["Adj Close"]].rename(columns={"Adj Close": "price"})
                df["ticker"] = ticker
                df["date"] = df.index
                df = df.reset_index(drop=True)
                
                # Compute daily returns
                df["return"] = df.groupby("ticker")["price"].pct_change()
                
                all_data.append(df)
                print(f"  ✓ {ticker}: {len(df)} observations")
            except Exception as e:
                warnings.warn(f"Error fetching {ticker}: {e}")
        
        if not all_data:
            raise ValueError("No ETF data fetched successfully.")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
        
        return combined

    def _align_signals(
        self,
        macro: pd.DataFrame,
        sentiment: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align macro, sentiment, and price data on common dates."""
        print("\nAligning signals across data sources...")
        
        # Prepare macro signals (monthly → daily via forward fill)
        macro = macro.copy()
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        macro = macro[["M_score"]].rename(columns={"M_score": "macro_signal"})
        macro = macro.resample("D").ffill()
        
        # Prepare sentiment signals (daily)
        sentiment = sentiment.copy()
        sentiment.index = pd.to_datetime(sentiment.index).tz_localize(None)
        
        # Use S_MA if available, otherwise S_raw
        if "S_MA" in sentiment.columns:
            sentiment = sentiment[["S_MA"]].rename(columns={"S_MA": "sentiment_signal"})
        else:
            sentiment = sentiment[["S_raw"]].rename(columns={"S_raw": "sentiment_signal"})
        
        # Merge all data
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        
        merged = prices.merge(macro, left_on="date", right_index=True, how="left")
        merged = merged.merge(sentiment, left_on="date", right_index=True, how="left")
        
        # Forward fill signals
        merged[["macro_signal", "sentiment_signal"]] = merged[
            ["macro_signal", "sentiment_signal"]
        ].fillna(method="ffill")
        
        # Drop rows with missing signals
        merged = merged.dropna(subset=["macro_signal", "sentiment_signal", "return"])
        
        print(f"  ✓ Aligned {len(merged)} observations across {merged['ticker'].nunique()} ETFs")
        
        return merged

    def _compute_fused_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fused signal s_t = w1·M_t + w2·S_t."""
        df = df.copy()
        
        df["fused_signal"] = (
            self.w_macro * df["macro_signal"] +
            self.w_sentiment * df["sentiment_signal"]
        )
        
        print(f"\n✓ Fused signal computed (w_macro={self.w_macro}, w_sentiment={self.w_sentiment})")
        
        return df

    def _estimate_expected_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate μ_t = α + β·s_t using rolling OLS.
        
        For each ticker, fit a rolling regression of next-day return on fused_signal.
        """
        print("\nEstimating expected returns via rolling OLS...")
        
        df = df.copy()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Create next-day return for regression
        df["return_next"] = df.groupby("ticker")["return"].shift(-1)
        
        results = []
        
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].copy()
            
            alpha_list = []
            beta_list = []
            mu_list = []
            
            for i in range(len(ticker_df)):
                if i < self.ols_window:
                    # Insufficient data for rolling window
                    alpha_list.append(np.nan)
                    beta_list.append(np.nan)
                    mu_list.append(np.nan)
                    continue
                
                # Extract rolling window
                window = ticker_df.iloc[i - self.ols_window : i]
                
                X = window["fused_signal"].values
                y = window["return_next"].values
                
                # Drop NaNs
                valid = ~(np.isnan(X) | np.isnan(y))
                X_valid = X[valid]
                y_valid = y[valid]
                
                if len(X_valid) < 30:
                    alpha_list.append(np.nan)
                    beta_list.append(np.nan)
                    mu_list.append(np.nan)
                    continue
                
                # OLS regression
                slope, intercept, _, _, _ = stats.linregress(X_valid, y_valid)
                
                # Predict for current fused_signal
                s_t = ticker_df.iloc[i]["fused_signal"]
                mu_t = intercept + slope * s_t
                
                # Cap at 5th and 95th percentiles
                mu_5 = np.percentile(y_valid, 5)
                mu_95 = np.percentile(y_valid, 95)
                mu_t = np.clip(mu_t, mu_5, mu_95)
                
                alpha_list.append(intercept)
                beta_list.append(slope)
                mu_list.append(mu_t)
            
            ticker_df["alpha"] = alpha_list
            ticker_df["beta"] = beta_list
            ticker_df["expected_return"] = mu_list
            
            results.append(ticker_df)
        
        df_with_mu = pd.concat(results, ignore_index=True)
        
        print(f"  ✓ Expected returns estimated for {df_with_mu['ticker'].nunique()} ETFs")
        
        return df_with_mu

    def _estimate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate σ_t (volatility) and VaR_95.
        """
        print("\nEstimating risk metrics (volatility, VaR)...")
        
        df = df.copy()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Rolling volatility
        df["volatility"] = df.groupby("ticker")["return"].transform(
            lambda x: x.rolling(window=self.vol_window, min_periods=5).std()
        )
        
        # VaR_95 = μ_t - 1.65·σ_t
        df["var_95"] = df["expected_return"] - 1.65 * df["volatility"]
        
        print(f"  ✓ Risk metrics computed")
        
        return df

    def _classify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime based on μ_t, σ_t, VaR_95, and signal confidence.
        """
        print("\nClassifying market regimes...")
        
        df = df.copy()
        
        # Compute confidence score
        df["confidence"] = self._compute_confidence(df)
        
        def classify_row(row):
            mu = row["expected_return"]
            vol = row["volatility"]
            var = row["var_95"]
            conf = row["confidence"]
            
            # High volatility → Unstable
            if pd.notna(vol) and vol > self.REGIME_THRESHOLDS["high_vol"]:
                return "Unstable"
            
            # Positive expected return and safe VaR → Risk-On
            if pd.notna(mu) and mu > self.REGIME_THRESHOLDS["mu_positive"]:
                if pd.notna(var) and var > self.REGIME_THRESHOLDS["var_threshold"]:
                    return "Risk-On"
            
            # Negative expected return with high confidence → Risk-Off
            if pd.notna(mu) and mu < self.REGIME_THRESHOLDS["mu_positive"]:
                if pd.notna(conf) and conf > self.REGIME_THRESHOLDS["high_confidence"]:
                    return "Risk-Off"
            
            # Default
            return "Neutral"
        
        df["regime"] = df.apply(classify_row, axis=1)
        
        print(f"  ✓ Regimes classified")
        print(f"    Regime distribution:")
        for regime, count in df["regime"].value_counts().items():
            print(f"      {regime}: {count} ({count/len(df)*100:.1f}%)")
        
        return df

    def _compute_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute confidence score C_t = (|s_t| / (|s_t| + k)) · exp(-σ_t / σ*).
        """
        s_abs = df["fused_signal"].abs()
        vol = df["volatility"]
        
        signal_term = s_abs / (s_abs + self.k_confidence)
        vol_term = np.exp(-vol / self.sigma_star)
        
        confidence = signal_term * vol_term
        
        return confidence

    def _generate_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio action recommendations based on regime."""
        print("\nGenerating portfolio recommendations...")
        
        df = df.copy()
        
        recommendation_map = {
            "Risk-On": "Increase credit exposure / extend duration",
            "Risk-Off": "Short duration or rotate to Treasuries",
            "Neutral": "Hold current positions / wait for confirmation",
            "Unstable": "Reduce position size / increase cash",
        }
        
        df["action_recommendation"] = df["regime"].map(recommendation_map)
        
        print(f"  ✓ Recommendations generated")
        
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("RISK & VALUATION AGENT - SUMMARY")
        print("=" * 60)
        
        print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Total observations: {len(df)}")
        print(f"ETFs analyzed: {', '.join(df['ticker'].unique())}")
        
        print("\n--- Expected Return Statistics ---")
        print(f"Mean μ_t: {df['expected_return'].mean():.4f}")
        print(f"Std μ_t:  {df['expected_return'].std():.4f}")
        print(f"Min μ_t:  {df['expected_return'].min():.4f}")
        print(f"Max μ_t:  {df['expected_return'].max():.4f}")
        
        print("\n--- Volatility Statistics ---")
        print(f"Mean σ_t: {df['volatility'].mean():.4f}")
        print(f"Std σ_t:  {df['volatility'].std():.4f}")
        
        print("\n--- Confidence Statistics ---")
        print(f"Mean confidence: {df['confidence'].mean():.4f}")
        print(f"High confidence (>0.75): {(df['confidence'] > 0.75).sum() / len(df) * 100:.1f}%")
        
        print("\n--- Value-at-Risk (95%) ---")
        print(f"Mean VaR_95: {df['var_95'].mean():.4f}")
        print(f"Min VaR_95:  {df['var_95'].min():.4f}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load signals from other agents
    if not os.path.exists("macro_signals.csv"):
        print("Error: macro_signals.csv not found. Run macro_agent.py first.")
        exit(1)
    
    if not os.path.exists("sentiment_signals.csv"):
        print("Error: sentiment_signals.csv not found. Run news_sentiment_agent.py first.")
        exit(1)
    
    macro_signals = pd.read_csv("macro_signals.csv", index_col=0, parse_dates=True)
    sentiment_signals = pd.read_csv("sentiment_signals.csv", index_col=0, parse_dates=True)
    
    # Initialize and run agent
    agent = RiskValuationAgent(
        w_macro=0.6,
        w_sentiment=0.4,
        return_window=60,
        vol_window=20,
        ols_window=756,
    )
    
    results = agent.run(
        macro_signals=macro_signals,
        sentiment_signals=sentiment_signals,
        etf_tickers=["LQD", "HYG", "IEF"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        output_path="risk_valuation_signals.csv",
    )
    
    print("\n✓ Risk & Valuation Agent completed successfully")