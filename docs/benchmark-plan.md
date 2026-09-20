# Ganache Engine benchmark plan

## 目的

モデルを先に固定せず、MacaronIMEの実際の入力軌跡で精度と遅延を比較する。

## 記録する値

- model id、revision、量子化、backend、GPU layers
- モデル初回ロード時間
- `/v1/createone` の p50 / p95 / p99 / 最大値
- Top-1、Top-K、候補分布の妥当性
- raw入力、文脈、英数字混在、固有名詞、typo、連続入力
- timeout、cancel、古いrevisionの応答破棄

## 比較条件

- 同じtrajectory、同じ候補数、同じcontext長
- cold startとwarm inferenceを分離
- CUDA / Vulkan / CPUを別集計
- モデル取得先のrevisionとSHA-256を固定

モデルweightsはcheckout内のgitignore対象`models/cache`に置く。CIや別checkoutで場所を変える場合は`GANACHE_MODEL_DIR`を使う。

Rakukanの`jinen-v1-xsmall-q5` CUDA実測値は、初期baselineとして記録する。現時点のbaselineは温間10件でp50 24ms、p95 32ms、最大32ms（候補取得完了まで）である。
