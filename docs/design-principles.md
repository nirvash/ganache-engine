# Ganache Engine 設計方針

## 位置づけ

Ganache は、低レイテンシな構造化予測を行うモデル非依存のローカル推論ランタイムである。特定のアプリケーションや入力方式のための変換エンジンではない。

```text
client task profile
  prompt / input / output schema / session
                ↓
Ganache Engine
  request lifecycle / cache / generation / resource control
                ↓
model backend
```

MacaronIME は最初の client であり、Ganache の設計上の親ではない。MacaronIME 固有の prompt、schema、入力正規化、候補の意味付けは MacaronIME 側に置く。

## Ganache が扱うもの

- model backend のロード、選択、runtime/backend 設定
- task prompt と任意の JSON input
- structured output schema
- session identity と session 状態
- generation parameters
- KV cache の保持・再利用・破棄
- timeout、cancel、request freshness、resource limits
- backend latency と resource usage の計測
- schema-constrained decoding と structured output validation

## Ganache が扱わないもの

- ローマ字、かな入力、読み仮名の生成
- IME の composition、候補表示、確定、TSF
- 日本語文章、候補、spam、タグ、NPC などのドメイン語彙
- 特定 client の prompt や output field の意味

たとえば `candidates[].text`、`candidates[].reading`、`candidates[].score` は MacaronIME が渡す schema の一例であり、Ganache core の型ではない。同じ API で分類、ランキング、proposal、行動選択などを扱える必要がある。

## API の境界

現在の最小契約は `POST /v1/predict` である。

```json
{
  "request_id": 1,
  "session_id": "composition-42",
  "prompt": "raw input と確定済み文脈から意図した文章を推測する",
  "input": {
    "raw": "nihongo",
    "context": ""
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "candidates": {"type": "array"}
    }
  },
  "generation": {
    "max_tokens": 64,
    "temperature": 0.0
  }
}
```

応答は `output` に structured value を返す。`request_id` と `session_id` は client が古い予測を破棄し、同一 session の状態を対応付けるための transport identity である。Ganache は値のドメイン上の意味を解釈しない。

## 実装段階

1. `PredictRequest/Response` と `/v1/predict` の汎用 contract
2. 単一 backend の prompt → 生成 → JSON value 化
3. output schema による validation と constrained decoding
4. session 単位の KV cache、cancel、timeout、freshness 制御
5. CUDA/Vulkan/CPU の backend 選択、resource telemetry、複数 model backend

現在の Jinen adapter は第2段階であり、JSON を生成できた場合はその値、JSON でない場合は文字列 value を返す。これは汎用 contract の疎通用であって、schema 適合を保証する最終実装ではない。
