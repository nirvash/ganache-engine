# Ganache Engine

MacaronIME の filling として使う、単独利用可能なモデル推論ライブラリおよび実行ラッパーです。

Ganache はモデル名を固定せず、モデル・量子化・CUDA/Vulkan/CPU backendを同一のIME評価で比較できることを目的にします。モデルweightsはこのrepositoryに含めず、`models/models.toml`で取得先と実行プロファイルを管理します。

## 初期構成

- `ganache-core`: 推論ライブラリと `/v1/createone` の共有型
- `ganache-service`: HTTP実行ラッパー（準備中）
- `models/models.toml`: モデル取得先・revision・runtime設定
- `docs/benchmark-plan.md`: モデル比較・精度・遅延の評価方針

## 開発

```powershell
cargo check --workspace
```

モデルweightsの取得、GPU backendの有効化、HTTP serviceの実装は後続の段階で追加します。通常のCIではモデルを取得しません。
