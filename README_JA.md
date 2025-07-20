# 転換点ブレイクアウト戦略 ― バックテスト

このリポジトリは、先に説明した「市場構造の転換点」を狙うハイブリッド型アルゴリズムの **最小検証プロトタイプ** を収録しています。

ワークフローは以下のとおりです。
1. Yahoo Finance から価格データ（OHLCV）を取得
2. テクニカル指標を計算
3. LightGBM / RandomForest で「転換確率モデル」を学習
4. 確率が閾値を超えた銘柄だけを監視対象に追加
5. ブレイクアウトを検出してエントリーし、
   * 損切り: ATR × α
   * 利確:   ATR × β
6. `backtesting.py` でバックテストを実行し、統計を出力

---
## 1. セットアップ
```bash
# Python 3.10 以上を推奨
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
LightGBM がインストールできない環境では、自動的に RandomForest へフォールバックします。

---
## 2. 実行方法
```bash
python turning_point_backtest.py \
  --symbol BTC-USD \
  --interval 1h \
  --start 2021-01-01 \
  --end   2024-01-01 \
  --train_split 0.7
```
* `symbol` … Yahoo Finance のティッカー（例: `AAPL`, `ETH-USD`）
* `interval` … データ頻度（`1h`, `30m`, `1d` など）
* 結果は `backtest_results/stats_<SYMBOL>.csv` に保存されます。

---
## 3. 主要ハイパーパラメータ
| パラメータ | 意味 | デフォルト |
|------------|------|------------|
| `prob_threshold` | 転換確率の下限値 | 0.7 |
| `breakout_lookback` | ブレイク判定に使う過去バー数 \(k\) | 20 |
| `atr_mult_sl` | 損切り幅（ATR × α） | 1.0 |
| `atr_mult_tp` | 利確幅（ATR × β） | 2.0 |
| `adx_exit` | ADX がこの値を下回ったら手仕舞い | 20 |

※ これらは `turning_point_backtest.py` 内の `TurningPointStrategy` クラス変数で設定されています。必要に応じて編集してください。

---
## 4. 注意事項（必読）
本スクリプトは **研究・検証用の最小構成** です。実運用前に、以下を必ずご確認ください。
* データフィードの信頼性と欠損処理
* 手数料・スリッページの厳密なモデリング
* ウォークフォワード／ライブフォワード検証の実施
* リスク管理（最大ドローダウン、ポジションサイズ制御 など）の強化

上記を怠ると、バックテスト結果と実運用成績が大きく乖離する恐れがあります。