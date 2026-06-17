# フォルダ構成の概要

## `nanochat/` — コアライブラリ

モデル・推論・学習に必要なすべての再利用可能なコンポーネントを含むメインパッケージ。

| ファイル | 役割 |
|---|---|
| `gpt.py` | GPTモデル定義 — RoPE・GQA・QK-norm・スライディングウィンドウアテンション・`relu²` MLP を備えたTransformerアーキテクチャ |
| `engine.py` | KVキャッシュを用いたバッチ推論エンジン。ツール呼び出し（Pythonカリキュレータ）とストリーミング生成をサポート |
| `common.py` | 共通ユーティリティ: dtype検出・DDP初期化/クリーンアップ・デバイス自動検出・`print0`・ピークFLOPSテーブル |
| `tokenizer.py` | 特殊トークン処理と会話レンダリングを備えたトークナイザーラッパー |
| `dataloader.py` | BOS整列ベストフィットパッキングを用いた分散データローダー |
| `dataset.py` | 事前学習データへのアクセス |
| `checkpoint_manager.py` | チェックポイントの保存・読み込み（モデル重み・オプティマイザ状態・メタデータ） |
| `optim.py` | `MuonAdamW` / `DistMuonAdamW` — 行列用Muon + 埋め込み/スカラー用AdamW を組み合わせたオプティマイザ |
| `flash_attention.py` | FA3統合とSDPAフォールバック |
| `fp8.py` | FP8学習サポート（`Float8Linear`） |
| `loss_eval.py` | Bits-per-byte (bpb) 検証損失の評価 |
| `report.py` | 学習ランのレポートログ |

---

## `scripts/` — 実行可能な学習・推論スクリプト

`nanochat/` ライブラリを組み合わせたエントリーポイントスクリプト群。`python -m scripts.<name>` または `torchrun` で実行する。

| ファイル | 役割 |
|---|---|
| `base_train.py` | **事前学習** — 生テキストからGPTをゼロから学習。スケーリング則・バッチサイズ自動計算・LR/モメンタム/重み減衰スケジュール・DDP・FP8・チェックポイントを処理 |
| `base_eval.py` | ベース（事前学習済み）モデルをCOREベンチマークで評価 |
| `chat_sft.py` | **教師あり微調整 (SFT)** — SmolTalk・MMLU・GSM8K・SpellingBeeなどのタスク混合でベースモデルを微調整 |
| `chat_rl.py` | **強化学習 (GRPO/REINFORCE)** — サンプリングされたロールアウトと報酬重み付き方策勾配を用いてGSM8KでRLファインチューニング |
| `chat_eval.py` | チャットモデルをChatCOREベンチマークで評価 |
| `chat_cli.py` | インタラクティブなCLIチャットインターフェース（シングルGPU・ストリーミング） |
| `chat_web.py` | Webベースのチャットインターフェース |
| `tok_train.py` | トークナイザーの学習 |
| `tok_eval.py` | トークナイザーの評価 |

---

## `tasks/` — ベンチマーク・学習タスク定義

各ファイルはデータセットを標準インターフェース（`__getitem__`・`evaluate`・`reward`）を持つ `Task` オブジェクトとしてラップする。学習スクリプト（SFTデータ混合）と評価スクリプト（ChatCORE）の両方で使用される。

| ファイル | タスク | 種別 |
|---|---|---|
| `common.py` | `Task` 基底クラス・`TaskMixture`・`TaskSequence`・`render_mc` ヘルパー | — |
| `gsm8k.py` | GSM8K 数学文章題（ツール呼び出しパース付き） | generative |
| `mmlu.py` | MMLU 57科目にわたる多肢選択問題 | categorical |
| `arc.py` | ARC-Easy / ARC-Challenge 理科問題 | categorical |
| `humaneval.py` | HumanEval コード生成 | generative |
| `smoltalk.py` | SmolTalk 汎用会話データセット | generative |
| `spellingbee.py` | スペリングタスク（`SimpleSpelling`・`SpellingBee`） | generative |
| `customjson.py` | `.jsonl` ファイルから任意の会話を読み込む | — |

---

## 全体フロー

```
tasks/          →  データセット + 評価基準の定義
nanochat/       →  モデル・オプティマイザ・エンジン・トークナイザー（ライブラリ）
scripts/        →  上記2つを使った学習・評価のオーケストレーション
```

典型的なパイプライン:

```
base_train.py（事前学習）→ chat_sft.py（SFT）→ chat_rl.py（RL）→ chat_cli.py / chat_web.py（推論）
```
