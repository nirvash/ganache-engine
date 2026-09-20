# Ganache Engine

MacaronIME の filling として使う、単独利用可能なモデル推論ライブラリおよび実行ラッパーです。

Ganache はモデル名を固定せず、モデル・量子化・CUDA/Vulkan/CPU backendを同一のIME評価で比較できることを目的にします。モデルweightsはこのrepositoryのgitignoreされた`models/cache`（または`GANACHE_MODEL_DIR`で指定した場所）に置き、`models/models.toml`で取得先と実行プロファイルを管理します。

## 初期構成

- `ganache-core`: 推論ライブラリと `/v1/createone` の共有型
- `ganache-models`: モデルregistryとcheckout内cache解決（weights取得は行わない）
- `ganache-service`: `/health` と `/v1/createone` のHTTP実行ラッパー（backend接続前の骨格）
- `models/models.toml`: モデル取得先・revision・runtime設定
- `docs/benchmark-plan.md`: モデル比較・精度・遅延の評価方針

## 開発

```powershell
cargo check --workspace
```

現段階のserviceはbackend未接続時に`503 backend is not available`を返します。モデルweightsの取得、GPU backendの有効化、実モデルadapterは後続の段階で追加します。通常のCIではモデルを取得しません。

モデル配置を変更する場合は`GANACHE_MODEL_DIR`を指定します。repository rootを明示する場合は`GANACHE_REPO_ROOT`を指定すると、その下の`models/cache`が使われます。

`ganache-models::Registry::verify_local`でモデルとtokenizerの存在・SHA-256を検証できます。registryにハッシュが未記載のbaselineは、ファイルが存在しても`verified = false`です。
