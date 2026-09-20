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

## 実測記録

2026-09-20時点で、checkout内`models/cache`の`jinen_v1_xsmall_q5`をGanache serviceへロードし、CPU backendで汎用`/v1/predict`を疎通した。task profileは`prompt = ニホンゴ`、`output_schema = string`、`max_tokens = 8`、10回連続実行で、全件`日本語`を返した。

- Ganache計測 latency: p50 8ms、最大9ms
- HTTP client wall time: p50 13ms、最大67ms（初回の1件を含む）
- GPU layers: 0（CPU実行）

これは実モデルをGanache経由で呼べることの確認であり、MacaronIMEの最終task profileの精度・CUDA性能・KV cache性能を示すものではない。次は同じprompt/schemaをCUDA backendで測定する。

同条件の再計測には、`ganache-service` の `ganache-bench` binary を使う。`cargo run -p ganache-service --bin ganache-bench` はCPU、`--features cuda` はCUDAを選択し、model load、warmup、p50/p95/maxを同じ形式で出力する。
