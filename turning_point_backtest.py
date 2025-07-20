import warnings
warnings.filterwarnings('ignore')

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# External libraries that may not be available by default
try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("yfinance is required. Install via `pip install yfinance`.")

try:
    from ta import momentum, trend, volatility
except ImportError:
    raise SystemExit("`ta` library is required. Install via `pip install ta`.")

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

try:
    from backtesting import Backtest, Strategy
except ImportError:
    raise SystemExit("backtesting.py is required. Install via `pip install backtesting`.")


def download_data(symbol: str, start: str = "2021-01-01", end: str = None, interval: str = "1h") -> pd.DataFrame:
    """Download OHLCV data via Yahoo Finance."""
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")

    # yfinance >=0.2 returns MultiIndex columns (field, ticker). Flatten.
    if isinstance(df.columns, pd.MultiIndex):
        # Keep only the first level (field) assuming single ticker requested
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append to DataFrame."""
    df = df.copy()

    # ATR
    atr_indicator = volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = atr_indicator.average_true_range()

    # RSI
    df["RSI"] = momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # ADX
    df["ADX"] = trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()

    # Bollinger Band width
    bb = volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["Close"]

    # Drop initial NaNs created by indicators
    df = df.dropna()
    return df


def create_labels(df: pd.DataFrame, future_window: int = 12, threshold: float = 1.3) -> pd.Series:
    """Create binary regime‐change labels based on future ATR / past ATR ratio."""
    future_atr = df["ATR"].shift(-future_window).rolling(future_window).mean()
    past_atr = df["ATR"].rolling(future_window).mean()
    ratio = future_atr / past_atr
    label = (ratio > threshold).astype(int)
    return label


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    if _LGB_AVAILABLE:
        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "ATR",
        "RSI",
        "ADX",
        "BB_width",
        "Close",
    ]
    return df[feature_cols]


class TurningPointStrategy(Strategy):
    # Class parameters (will be overridable via Backtest)
    prob_threshold: float = 0.7  # P(turning_point) threshold
    breakout_lookback: int = 20  # k bars lookback for breakout
    atr_mult_sl: float = 1.0  # Stop loss ATR multiple
    atr_mult_tp: float = 2.0  # Take profit ATR multiple
    adx_exit: float = 20  # Exit if ADX below this

    def init(self):
        # Pre-compute numpy arrays for speed
        self.atr = self.data.ATR
        self.adx = self.data.ADX
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low
        self.prob = self.data.Prob  # predicted probability

    def next(self):
        i = len(self.data) - 1  # current index
        atr_now = self.atr[i]
        if np.isnan(atr_now):
            return

        if not self.position:
            # Check probability condition first
            if self.prob[i] < self.prob_threshold:
                return

            # Breakout detection
            k = self.breakout_lookback
            if i < k:
                return
            high_max = self.high[i - k:i].max()
            low_min = self.low[i - k:i].min()

            price = self.close[i]
            if price > high_max:
                sl = price - self.atr_mult_sl * atr_now
                tp = price + self.atr_mult_tp * atr_now
                self.buy(sl=sl, tp=tp)
            elif price < low_min:
                sl = price + self.atr_mult_sl * atr_now
                tp = price - self.atr_mult_tp * atr_now
                self.sell(sl=sl, tp=tp)
        else:
            # Optional early exit if ADX falls
            if self.adx[i] < self.adx_exit:
                self.position.close()


def run_backtest(
    df: pd.DataFrame,
    cash: float = 100_000,
    commission: float = 0.0005,
    ) -> dict:
    bt = Backtest(
        df,
        TurningPointStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True,
    )
    stats = bt.run()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Turning‐Point Regime‐Change Breakout Backtest")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol (Yahoo Finance)")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g., 1h, 30m, 1d)")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train/test split ratio")

    args = parser.parse_args()

    # 1. Data download
    print(f"Downloading data for {args.symbol}…")
    df = download_data(args.symbol, start=args.start, end=args.end, interval=args.interval)

    # 2. Feature engineering
    df = add_indicators(df)

    # 3. Label creation
    df["Label"] = create_labels(df)

    # 4. Train/Test split
    df = df.dropna()
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = build_feature_matrix(train_df)
    y_train = train_df["Label"]
    X_test = build_feature_matrix(test_df)

    # 5. Model training
    print("Training model…")
    model = train_model(X_train, y_train)

    # 6. Prediction on full dataset
    full_X = build_feature_matrix(df)
    prob_pred = pd.Series(model.predict_proba(full_X)[:, 1], index=df.index)
    df["Prob"] = prob_pred

    # 7. Prepare DataFrame for Backtest (must include required OHLCV columns)
    bt_df = df.copy()
    # The Backtesting library expects columns: Open, High, Low, Close, Volume plus any custom.

    # 8. Run Backtest
    print("Running backtest…")
    stats = run_backtest(bt_df)

    # 9. Output results
    print(stats)

    # 10. Save results
    out_dir = Path("backtest_results")
    out_dir.mkdir(exist_ok=True)
    stats.to_csv(out_dir / f"stats_{args.symbol}.csv")
    print(f"Statistics saved to {out_dir / f'stats_{args.symbol}.csv'}")


if __name__ == "__main__":
    main()