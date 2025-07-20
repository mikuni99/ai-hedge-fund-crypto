# Turning-Point Breakout Strategy – Backtest

This repository contains a minimal, end-to-end prototype of the "市場構造の転換点" ハイブリッド型アルゴリズム described earlier.

The workflow is:
1. 価格データ取得（Yahoo Finance経由）
2. テクニカル指標生成
3. 転換確率モデル（LightGBM / RandomForest）学習
4. 確率がしきい値を超えた銘柄のみ監視
5. ブレイクアウト検出でエントリー、ATR×α で損切り / ATR×β で利確
6. `backtesting.py` で検証 & 統計出力

## 1. セットアップ
```bash
# Python 3.10+ 推奨
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
LightGBM がインストールできない環境では、自動的に RandomForest へフォールバックします。

## 2. 実行方法
```bash
python turning_point_backtest.py \
  --symbol BTC-USD \
  --interval 1h \
  --start 2021-01-01 \
  --end   2024-01-01 \
  --train_split 0.7
```
‐ `symbol` は Yahoo Finance 形式（例: *AAPL*, *ETH-USD*）。
‐ `interval` は `1h`, `30m`, `1d` など。
‐ 出力例は `backtest_results/stats_<SYMBOL>.csv` に保存されます。

## 3. 主要ハイパーパラメータ
| パラメータ | 意味 | デフォルト |
|------------|------|------------|
| `prob_threshold` | P(turning_point) の下限値 | 0.7 |
| `breakout_lookback` | ブレイク判定過去バー数 k | 20 |
| `atr_mult_sl` | 損切り幅 (ATR×α) | 1.0 |
| `atr_mult_tp` | 利確幅 (ATR×β) | 2.0 |
| `adx_exit` | ADX 低下で手仕舞う閾値 | 20 |

`turning_point_backtest.py` 内で `TurningPointStrategy` クラス変数として定義されています。必要に応じて書き換えてください。

---
本スクリプトは **検証用の最小構成** です。実運用前には、
* データフィードの確実性、
* 手数料・スリッページの精緻化、
* ウォークフォワード／ライブフォワードの実施、
* リスク管理ルールの強化、
などを十分に行ってください。