# Ganache Engine benchmark plan

## 目的

モデルを先に固定せず、各クライアントの task profile と入力軌跡で精度と遅延を比較する。MacaronIME の軌跡は最初の評価セットであり、Ganache の API や評価対象を IME に限定しない。

## 記録する値

- model id、revision、量子化、backend、GPU layers
- モデル初回ロード時間
- `/v1/predict` の p50 / p95 / p99 / 最大値
- schema 適合率、出力妥当性、task 固有の精度
- input、prompt、session、連続リクエスト
- timeout、cancel、古いrevisionの応答破棄

## 比較条件

- 同じtrajectory、同じ候補数、同じcontext長
- cold startとwarm inferenceを分離
- CUDA / Vulkan / CPUを別集計
- モデル取得先のrevisionとSHA-256を固定

モデルweightsはcheckout内のgitignore対象`models/cache`に置く。CIや別checkoutで場所を変える場合は`GANACHE_MODEL_DIR`を使う。

Rakukanの`jinen-v1-xsmall-q5` CUDA実測値は、MacaronIME task profile の初期baselineとして記録する。現時点のbaselineは温間10件でp50 24ms、p95 32ms、最大32ms（候補取得完了まで）であり、Ganache の汎用ランタイム全体の性能値ではない。
