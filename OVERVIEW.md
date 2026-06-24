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

### 主要ファイルの詳細

#### `gpt.py` — GPTモデル定義

`GPTConfig` データクラスで `depth`（`n_layer`）を起点にアーキテクチャ全体をパラメータ化する。主なコンポーネント:

- **`CausalSelfAttention`**: RoPE（回転位置埋め込み）・QK-norm（クエリ/キーの正規化 + スケーリング `*1.2`）・GQA（グループクエリアテンション）を実装。`window_pattern` 文字列（例: `"SSSL"`）に基づき、レイヤーごとにフルコンテキスト(`L`) またはスライディングウィンドウ(`S`, コンテキスト長の1/4) を適用する。推論時は `KVCache` を用いたインクリメンタルデコードに対応。
- **`MLP`**: `relu²`（ReLU → 二乗）活性化関数を使用。隠れ層の次元は `4 * n_embd`。
- **`Block`**: Pre-norm 構成（`norm(x)` → Attention → 残差加算 → `norm(x)` → MLP → 残差加算）。
- **`GPT`**: モデル本体。以下の特殊機構を持つ:
  - **Smear**: 前トークンの埋め込みを現トークンに混合するゲート機構（安価なバイグラム情報の注入）。
  - **Backout**: 中間レイヤー（`n_layer // 2`）の残差を最終出力から減算し、低レベル特徴を除去。
  - **Value Embeddings**: ResFormer スタイルの値埋め込み。交互のレイヤーに配置し、入力依存のゲートでアテンションの `v` に加算。
  - **`resid_lambdas` / `x0_lambdas`**: レイヤーごとの学習可能スカラー。残差ストリームのスケーリングと初期埋め込みのブレンドを制御。
  - **Logit softcap**: ロジットを `tanh(logits/15)*15` でソフトクリッピング。
  - `init_weights()` でメタデバイス上のモデルパラメータを一括初期化（Embedding: `N(0, 0.8)`, 射影: ゼロ初期化, Attention Q/K/V: 一様分布 `U(-s, s)` where `s = √3 / √n_embd`）。
  - `setup_optimizer()` でパラメータを種別ごとにグルーピングし、`MuonAdamW`（シングルGPU）または `DistMuonAdamW`（分散）を構築。学習率は `1/√d_model` でスケール。

#### `engine.py` — 推論エンジン

`Engine` クラスが `generate()` ジェネレータを提供。主な機能:

- **Prefill → Decode 分離**: バッチサイズ1でプロンプトをプリフィルし、KVキャッシュを `num_samples` 分に複製して並列デコード。
- **KVCache**: FA3 の `flash_attn_with_kvcache` API に最適化。テンソルレイアウトは `(B, T, H, D)`。`cache_seqlens` で位置を追跡し、インプレースでキャッシュを更新。
- **ツール呼び出しステートマシン**: 各行（`RowState`）が独立に `<|python_start|>` / `<|python_end|>` トークンを監視。Python式を検出すると `use_calculator()` で安全に評価し、結果を `<|output_start|>...<|output_end|>` として強制注入（`forced_tokens` キュー）。数式と `.count()` メソッドのみ許可し、危険なパターンはブロック。
- **サンプリング**: Top-k フィルタリングと温度パラメータに対応。`temperature=0` で貪欲デコード。

#### `optim.py` — MuonAdamW オプティマイザ

2種類のパラメータに対して異なる最適化アルゴリズムを適用する統合オプティマイザ:

- **AdamW** (埋め込み・スカラーパラメータ用): `torch.compile` でフューズされたカーネル `adamw_step_fused()` を使用。重み減衰 → モメンタム更新 → バイアス補正 → パラメータ更新を1グラフで実行。0次元CPUテンソルでハイパーパラメータを渡すことで再コンパイルを回避。
- **Muon** (2D行列パラメータ用): SGDモメンタムの勾配を **Polar Express** アルゴリズム（Newton-Schulz反復の改良版, 5ステップ）で直交化。その後 **NorMuon** 分散削減（列/行ごとの適応的学習率正規化）と **Cautious** 重み減衰（勾配とパラメータの符号が一致する要素のみ減衰適用）を行う。すべて `muon_step_fused()` で1カーネルに融合。
- **`DistMuonAdamW`** (分散版): 3フェーズ非同期通信（reduce → compute → gather）でオーバーラップを最大化。AdamW は ZeRO-2 スタイルでオプティマイザ状態をシャーディング。Muon は全パラメータをスタックし、ランクごとにチャンク分割して `reduce_scatter` → 更新 → `all_gather`。

#### `tokenizer.py` — トークナイザー

GPT-4スタイルのBPEトークナイザー。2つの実装を提供:

- **`HuggingFaceTokenizer`**: HuggingFace Tokenizers ライブラリベース。学習（`train_from_iterator`）と推論の両方に対応。GPT-4 の分割パターン（ただし数字は `\p{N}{1,2}` に変更、小さいボキャブラリサイズに最適化）を使用。
- **`RustBPETokenizer`**: 学習には `rustbpe`（Rust実装）、推論には `tiktoken` を使用する高速実装。実運用ではこちらがデフォルト。
- **特殊トークン**: `<|bos|>`, `<|user_start|>/<|user_end|>`, `<|assistant_start|>/<|assistant_end|>`, `<|python_start|>/<|python_end|>`, `<|output_start|>/<|output_end|>` の9種。
- **`render_conversation()`**: 会話データ（messages配列）をトークン列とロスマスクに変換。ユーザー発話・特殊トークン・ツール出力は `mask=0`（学習対象外）、アシスタント発話は `mask=1`（学習対象）。システムメッセージはユーザーメッセージに結合。
- **`render_for_completion()`**: RL用。会話の最後のアシスタントメッセージを削除し、`<|assistant_start|>` を末尾に付与してモデルに補完を促す。

#### `dataloader.py` — 分散データローダー

事前学習用の **BOS整列ベストフィットパッキング** データローダー:

- **パッキングアルゴリズム**: バッファ内の文書から「行に完全に収まる最大の文書」を繰り返し選択。どの文書も収まらなくなった場合、最短の文書をクロップして残りスペースを埋める。パディングなし（100%利用率）、約35%のトークンがクロップにより破棄。
- **BOS整列**: すべての行が `<|bos|>` トークンで始まる。各トークンが常にBOSまでの完全なコンテキストを参照できることを保証。
- **分散対応**: `_document_batches()` がParquetファイルをDDPランクごとにシャーディング。`resume_state_dict`（`pq_idx`, `rg_idx`, `epoch`）による中断再開をサポート。
- **メモリ最適化**: ピン留めメモリを使った事前割り当てバッファ（`cpu_buffer` → `gpu_buffer`）と単一の HtoD 転送で効率化。

#### `flash_attention.py` — Flash Attention 統合

FA3（Flash Attention 3）の API と完全互換のインターフェースを公開し、ハードウェアに応じて自動的に実装を切り替える:

- **FA3** (Hopper GPU, SM90, bf16 のみ): `kernels` パッケージから動的にロード。学習時は `flash_attn_func()`、推論時は `flash_attn_with_kvcache()` を使用。
- **SDPA フォールバック** (Ampere以前, Blackwell, CPU, MPS): PyTorch の `F.scaled_dot_product_attention` を使用。スライディングウィンドウはブールマスクで明示的に実装。GQA にも対応（`enable_gqa=True`）。
- テンソルレイアウトは `(B, T, H, D)` で統一し、SDPA 使用時のみ内部で `(B, H, T, D)` に転置。

#### `checkpoint_manager.py` — チェックポイント管理

モデル・オプティマイザ・メタデータの保存と読み込みを管理:

- **保存**: `save_checkpoint()` がモデルパラメータ（`model_{step}.pt`）、メタデータ（`meta_{step}.json`）をランク0で、オプティマイザ状態（`optim_{step}_rank{rank}.pt`）を各ランクで保存。オプティマイザ状態は分散学習時にランクごとにシャーディング。
- **読み込み**: `build_model()` がメタデータからモデル設定を復元し、メタデバイス上でモデルを構築 → 対象デバイスに移動 → 重みをロード。`torch.compile` のプレフィックス `_orig_mod.` の除去や、旧チェックポイントに欠けるキー（`window_pattern`, `resid_lambdas`, `x0_lambdas`）のパッチも自動処理。
- **自動検出**: `load_model(source)` で `source` を `"base"` / `"sft"` / `"rl"` で指定すると、対応するチェックポイントディレクトリから最大のモデル（`d{N}` 形式で最大の深さ）・最新のステップを自動選択。

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

### 主要ファイルの詳細

#### `base_train.py` — 事前学習オーケストレーション

スケーリング則に基づいて学習ハイパーパラメータを自動計算し、事前学習の全工程を管理する:

- **学習ホライズンの自動計算**: `--target_param_data_ratio` から目標データ量 `D` を算出。Power Lines 論文に基づきバッチサイズを `D^0.383` でスケーリング。基準モデル d12 (`B_REF = 2^19 = 524,288 tokens`) を基準に、学習率 `η ∝ √(B/B_ref)`、重み減衰 `λ = λ_ref · √(B/B_ref) · (D_ref/D)` を自動調整。
- **スケジュール**: LR は warmup → cosine decay → cooldown の3段階。モメンタムは初期値からピーク値へのウォームアップ。
- **学習ループ**: 勾配累積による効率的なマイクロバッチ処理。`torch.compile` で高速化されたモデルの `forward` を使用し、FP8 学習時は `Float8Linear` モジュールのスケール動的調整も実施。評価時は `disable_fp8()` コンテキストマネージャでFP8を無効化。
- **評価**: 定期的に検証データで BPB (bits-per-byte) を計測し、CORE メトリクス（ARC, MMLU, HumanEvalなどの総合スコア）で進捗を追跡。

#### `chat_sft.py` — 教師あり微調整 (SFT)

事前学習済みチェックポイントからロードし、タスク混合データで微調整する:

- **データ混合**: SmolTalk（汎用会話）、MMLU（多肢選択）、GSM8K（数学）、SpellingBee などの `TaskMixture` を構成。MMLU と GSM8K のエポック数は CLI で設定可能。
- **BOS整列ベストフィットパッキング**: `sft_data_generator_bos_bestfit()` が会話をトークン化し、ベストフィットアルゴリズムで行に詰める。アシスタント以外のトークン（ユーザー発話・特殊トークン・パディング）は `ignore_index=-1` でマスク。
- **オプティマイザの温スタート**: 事前学習チェックポイントからモメンタムバッファを継承し、学習率のみリセット。LR スケジュールは進捗率 (0→1) ベース（データセットサイズに依存しない自動停止）。
- **ChatCORE評価**: ARC、MMLU、GSM8K、HumanEval、SpellingBee の正答率で微調整の効果を追跡。

#### `chat_rl.py` — 強化学習 (GRPO)

GSM8K 上で簡略化された GRPO（Group Relative Policy Optimization）を実行:

- **オンポリシー**: 毎ステップ現在のモデルでロールアウトを生成し、即座に方策勾配を計算。KL正則化・PPOの比率クリッピングは不使用（オンポリシーのため不要）。
- **ロールアウト生成**: プロンプトから `num_samples` 個のサンプルを生成 → 回答をデコード → `reward()` で正誤を判定（0 or 1）。
- **アドバンテージ計算**: `advantage = reward - mean(rewards)` の単純な平均ベースライン（z-score正規化は不使用）。トークンレベルで正規化し、方策勾配を計算。
- **評価**: pass@k（k個サンプル中1つ以上正解なら成功）を GSM8K テストセットで計測。分散環境では `all_reduce` で集約。

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

### 主要ファイルの詳細

#### `common.py` — タスク基盤

すべてのベンチマークの基底クラスと合成ユーティリティを定義:

- **`Task`**: 軽量なデータセットビュー。`start`, `stop`, `step` でスライスでき、`__getitem__()` で会話形式のサンプルを、`evaluate()` でモデル出力の採点を、`reward()` で RL 用の報酬（float）を返す。サブクラスは `_items` リストを構築するだけで良い。
- **`TaskMixture`**: 複数の `Task` を結合し、決定論的にシャッフル。SFT のデータ混合に使用。
- **`TaskSequence`**: タスクを順番に連結（カリキュラム学習用）。
- **`render_mc()`**: 多肢選択問題を `A) ... B) ... C) ...` 形式のテキストにレンダリングするヘルパー。

#### `gsm8k.py` — GSM8K 数学文章題

GSM8K データセットをラップし、ツール呼び出し（`<< expression = result >>`）形式での解答生成と自動評価を提供:

- `__getitem__()`: 質問と解答をユーザー/アシスタントの会話形式に変換。解答中の `<<expr>>` パターンを `<|python_start|>expr<|python_end|><|output_start|>result<|output_end|>` に変換。
- `evaluate()`: モデル出力から `####` マーカー以降の数値を抽出し、カンマ除去・正規化後に正解と比較。
- `reward()`: `evaluate()` の結果を 0.0 / 1.0 のfloatで返す（RL用）。

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
