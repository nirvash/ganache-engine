# Ganache Engine

低レイテンシな構造化予測を行う、モデル非依存のローカル推論ランタイムです。

Ganache はアプリケーションの意味論を持ちません。クライアントが `prompt`、`input`、`output_schema`、必要なら `session_id` と生成パラメータを渡し、モデル backend がその task に対する構造化出力を返します。MacaronIME は最初のクライアントにすぎず、Ganache の用途・ドメイン・入力方式を規定しません。

```text
MacaronIME
    ↓ task profile: prompt / schema / session / input
Ganache Engine
    ↓ model backend
local model runtime
```

## 初期構成

- `ganache-core`: モデル非依存の structured prediction contract
- `ganache-models`: モデル registry と checkout 内 cache 解決（weights 取得は行わない）
- `ganache-jinen`: Jinen GGUF model backend（比較用の最初の adapter、現段階は greedy generation）
- `ganache-service`: `/health` と `/v1/predict` の HTTP 実行ラッパー
- `models/models.toml`: モデル取得先・revision・runtime 設定
- `docs/benchmark-plan.md`: task profile ごとのモデル比較・精度・遅延の評価方針

## API contract

`POST /v1/predict` は次の汎用リクエストを受け取ります。

```json
{
  "request_id": 1,
  "session_id": "optional-session",
  "prompt": "classify the input",
  "input": {"text": "..."},
  "output_schema": {"type": "object"},
  "generation": {"max_tokens": 64, "temperature": 0.0},
  "metadata": {}
}
```

レスポンスの `output` は schema に従う structured value です。たとえば MacaronIME は task profile 側で `candidates[].text`、`candidates[].reading`、`candidates[].score` を定義できますが、そのフィールドや「候補」という概念は Ganache core の責務ではありません。

同じ contract は、spam 判定、タグ分類、選択肢ランキング、open-set proposal、NPC の次行動選択、UI 操作候補などにも利用できます。

## 開発

```powershell
cargo check --workspace
```

モデルを配置した状態で、Jinen比較用の同一条件ベンチマークを実行できます。

```powershell
$env:GANACHE_REPO_ROOT = (Get-Location).Path
cargo run -p ganache-service --bin ganache-bench
# CUDA版:
cargo run -p ganache-service --bin ganache-bench --features cuda
```

`GANACHE_GPU_LAYERS`、`GANACHE_MAIN_GPU`、`GANACHE_BENCH_SAMPLES`、`GANACHE_BENCH_WARMUPS`で条件を変更できます。ベンチマークは`PredictRequest`を直接実行するため、HTTPやIME UIの待ち時間を含まないbackend比較値です。

現段階の service はモデル weights が未配置なら `503 backend is not available` を返します。registry の Jinen profile は、Ganache の汎用 contract とモデル/backend の比較経路を検証するための最初の adapter です。Jinen backend は渡された prompt を greedy 生成し、JSON を生成できた場合は JSON value、そうでない場合は JSON string として返します。別モデルや別 backend は同じ `InferenceBackend` contract に追加します。schema-constrained decoding、KV cache、追加 backend、モデル weights の取得は後続の段階で追加します。通常の CI ではモデルを取得しません。

Jinen backend を含む Windows ビルドでは Visual Studio 2022 の CMake generator を使います。CUDA 版は `cargo test --workspace --features ganache-service/cuda` など、service 側 feature から有効化します。

モデル配置を変更する場合は `GANACHE_MODEL_DIR` を指定します。repository root を明示する場合は `GANACHE_REPO_ROOT` を指定すると、その下の `models/cache` が使われます。

`ganache-models::Registry::verify_local` でモデルと tokenizer の存在・SHA-256 を検証できます。モデル本体は repository に commit しません。
