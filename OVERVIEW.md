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

<details>
<summary><b><code>gpt.py</code> — GPTモデル定義</b></summary>

`GPTConfig` の `depth`（= `n_layer`）を起点に、モデル全体の次元・ヘッド数・語彙サイズなどが自動で決まる。

#### レイヤー構成

```
入力トークン
  │
  ▼
Embedding (wte)  ──→  norm  ──→  Smear（前トークンの混合）
  │
  ▼  ×n_layer
┌──────────────────────────────────────────┐
│  resid_lambda * x  +  x0_lambda * x0    │  ← 残差スケーリング + 初期埋め込みブレンド
│       │                                  │
│       ▼                                  │
│  CausalSelfAttention                     │
│    - QK-norm（*1.2 スケーリング）          │
│    - RoPE（回転位置埋め込み）              │
│    - GQA（グループクエリアテンション）       │
│    - window_pattern で窓幅を制御          │
│    - Value Embeddings（交互レイヤー）      │
│       │                                  │
│       ▼                                  │
│  MLP（relu² 活性化, 隠れ層 = 4 * d_model）│
└──────────────────────────────────────────┘
  │
  ▼
Backout（中間レイヤーの残差を減算）
  │
  ▼
lm_head  ──→  softcap: tanh(logits/15) * 15
```

#### 特殊機構

| 機構 | 概要 |
|---|---|
| **Smear** | 前トークンの埋め込みをゲート付きで現トークンに加算。バイグラム情報を安価に注入する |
| **Backout** | `n_layer // 2` 地点の中間表現を最終出力から学習可能な係数で減算。低レベル特徴がロジットに漏れるのを防ぐ |
| **Value Embeddings** | ResFormer スタイル。トークンIDから直接埋め込みを引き、アテンションの `v` に加算する。交互のレイヤーのみに配置 |
| **`resid_lambdas`** | レイヤーごとの残差スケーリング係数（学習可能スカラー） |
| **`x0_lambdas`** | 初期埋め込み `x0` をどの程度各レイヤーにブレンドするかを制御するスカラー |
| **Logit softcap** | ロジットの値域を `[-15, 15]` に滑らかに制限し、学習を安定化する |

#### 重みの初期化

| パラメータ | 初期化方法 |
|---|---|
| Embedding | `N(0, 0.8)` |
| Q / K / V | 一様分布 `U(-s, s)`, `s = √3 / √n_embd` |
| 出力射影 | ゼロ初期化 |

`setup_optimizer()` はパラメータを「行列」「埋め込み」「スカラー」に分類し、
行列には Muon、それ以外には AdamW を適用するオプティマイザを構築する。
学習率は `1/√d_model` でスケールされる。

</details>

<details>
<summary><b><code>engine.py</code> — 推論エンジン</b></summary>

`Engine.generate()` がプロンプトを受け取り、トークンを1つずつ生成するジェネレータ。

#### 生成フロー

```
プロンプト（トークン列）
  │
  ▼
① Prefill: バッチサイズ1でプロンプト全体を処理し、KVキャッシュを構築
  │
  ▼
② Clone: KVキャッシュを num_samples 分に複製
  │
  ▼
③ Decode ループ（1トークンずつ）:
     モデルに1トークン入力 → ロジット取得 → サンプリング → キャッシュ更新
     │
     └─ ツール呼び出し検出時:
          <|python_start|> ... <|python_end|> の間の式を評価し、
          結果を <|output_start|> ... <|output_end|> として強制注入
```

#### KVCache

FA3 の `flash_attn_with_kvcache` API に合わせたレイアウト `(B, T, H, D)` を使用。
`cache_seqlens` テンソルで各行の現在位置を追跡し、キャッシュはインプレースで更新される。

#### ツール呼び出し

各行が独立した `RowState`（ステートマシン）を持つ。
モデルが `<|python_start|>` を出力すると式の蓄積を開始し、`<|python_end|>` で `use_calculator()` を呼ぶ。
安全性のため、許可されるのは **数式** と **`.count()` メソッド** のみ。

</details>

<details>
<summary><b><code>optim.py</code> — MuonAdamW オプティマイザ</b></summary>

パラメータの形状に応じて 2 種類のアルゴリズムを自動で使い分ける統合オプティマイザ。

#### パラメータの分類と適用アルゴリズム

| パラメータの種類 | 例 | アルゴリズム |
|---|---|---|
| 2D 行列 | Attention の Q/K/V/O, MLP の重み | **Muon** |
| 埋め込み・スカラー | `wte`, `lm_head`, `resid_lambdas` 等 | **AdamW** |

#### Muon の更新ステップ

```
勾配 g
  │
  ▼
① Nesterov モメンタム: buf = momentum * buf + g,  g' = g + momentum * buf
  │
  ▼
② Polar Express 直交化（Newton-Schulz 反復 × 5ステップ）
   → 勾配行列を直交行列に変換し、すべての特異値を均一化
  │
  ▼
③ NorMuon: 列/行ごとに RMS を計算し、適応的にスケーリング
   → パラメータごとの学習率のばらつきを抑制
  │
  ▼
④ Cautious 重み減衰: 勾配とパラメータの符号が一致する要素のみ減衰
```

上記すべてが `muon_step_fused()` として `@torch.compile` で1カーネルに融合される。
AdamW 側も同様に `adamw_step_fused()` で融合。
ハイパーパラメータは **0次元CPUテンソル** で渡し、値の変更による再コンパイルを回避している。

#### 分散版 (`DistMuonAdamW`)

通信と計算をオーバーラップさせる 3 フェーズ構成:

```
Phase 1: reduce_scatter（勾配の集約）
Phase 2: 各ランクが担当チャンクを更新
Phase 3: all_gather（更新済みパラメータの配布）
```

AdamW は ZeRO-2 スタイルでオプティマイザ状態をランクごとにシャーディングする。

</details>

<details>
<summary><b><code>tokenizer.py</code> — トークナイザー</b></summary>

GPT-4 スタイルの BPE トークナイザー。2 つの実装がある:

| 実装 | 学習 | 推論 | 用途 |
|---|---|---|---|
| `HuggingFaceTokenizer` | HuggingFace Tokenizers | 同左 | 実験用 |
| `RustBPETokenizer` | `rustbpe`（Rust） | `tiktoken` | **デフォルト（高速）** |

#### 特殊トークン（9種）

```
<|bos|>
<|user_start|>     <|user_end|>
<|assistant_start|> <|assistant_end|>
<|python_start|>   <|python_end|>
<|output_start|>   <|output_end|>
```

#### 会話のレンダリング

`render_conversation()` は messages 配列をトークン列とロスマスクに変換する。

```
<|bos|> <|user_start|> こんにちは <|user_end|> <|assistant_start|> はい！ <|assistant_end|>
mask:  0        0          0          0             0              1       0
```

- `mask = 1`: アシスタント発話（学習対象）
- `mask = 0`: それ以外（ユーザー発話・特殊トークン・ツール出力）

`render_for_completion()` は RL 用。最後のアシスタント応答を削除し、
`<|assistant_start|>` を末尾に付けてモデルに続きを生成させる。

</details>

<details>
<summary><b><code>dataloader.py</code> — 分散データローダー</b></summary>

事前学習用の **BOS整列ベストフィットパッキング** データローダー。

#### パッキングの仕組み

1行（= シーケンス長）に複数の文書を詰め込む。パディングは一切使わない。

```
行の空き容量: ████████████████████░░░░░░░░  (残り 8 トークン)

① バッファから「空きに完全に収まる最大の文書」を選ぶ
② 収まる文書がなくなったら、最短の文書をクロップして残りを埋める
③ すべての行は <|bos|> で始まる（BOS整列）
```

- パディングなし → **100%の利用率**
- 約35%のトークンがクロップにより破棄されるが、BOS整列の恩恵（各トークンが常に完全なコンテキストを持つ）が上回る

#### 分散・再開

- Parquet ファイルを DDP ランクごとにシャーディング
- `state_dict`（`pq_idx`, `rg_idx`, `epoch`）を保存すれば、中断した位置から再開可能
- ピン留めメモリの事前割り当てバッファで HtoD 転送を最適化

</details>

<details>
<summary><b><code>flash_attention.py</code> — Flash Attention 統合</b></summary>

FA3（Flash Attention 3）の API と互換のインターフェースを公開し、
ハードウェアに応じて実装を自動的に切り替える。

| 条件 | 使用される実装 |
|---|---|
| Hopper GPU（SM90）+ bf16 | **FA3** — `kernels` パッケージから動的ロード |
| それ以外（Ampere, Blackwell, CPU, MPS） | **SDPA** — `F.scaled_dot_product_attention` |

公開 API は 2 つ:

- `flash_attn_func(q, k, v)` — 学習用（KVキャッシュなし）
- `flash_attn_with_kvcache(q, k_cache, v_cache, ...)` — 推論用（KVキャッシュあり）

テンソルレイアウトは `(B, T, H, D)` で統一。SDPA 使用時は内部で `(B, H, T, D)` に転置する。
スライディングウィンドウはブールマスクで実装し、GQA にも対応。

</details>

<details>
<summary><b><code>checkpoint_manager.py</code> — チェックポイント管理</b></summary>

モデル・オプティマイザ・メタデータの保存と読み込みを一元管理する。

#### ファイル構成

```
checkpoints/
  └── d12/                     ← モデルタグ（depth=12）
        ├── model_001000.pt    ← モデルパラメータ（ランク0が保存）
        ├── meta_001000.json   ← ハイパーパラメータ・設定
        ├── optim_001000_rank0.pt  ← オプティマイザ状態（各ランクが保存）
        └── optim_001000_rank1.pt
```

#### 読み込み時の自動処理

- `torch.compile` が付与するプレフィックス `_orig_mod.` を自動除去
- 旧チェックポイントに不足するキー（`window_pattern`, `resid_lambdas`, `x0_lambdas`）をデフォルト値でパッチ
- CPU / MPS 環境では bf16 → fp32 に自動変換

#### 便利関数

`load_model("base" | "sft" | "rl")` を呼ぶだけで、
対応するディレクトリから最大の深さのモデル・最新ステップを自動選択してロードできる。

</details>

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

<details>
<summary><b><code>base_train.py</code> — 事前学習オーケストレーション</b></summary>

スケーリング則に基づいてハイパーパラメータを自動計算し、事前学習を管理する。

#### ハイパーパラメータの自動計算

基準モデル d12（`B_REF = 2^19 = 524,288 tokens`）からの比率でスケーリングする:

| パラメータ | 計算方法 |
|---|---|
| データ量 `D` | `--target_param_data_ratio` × パラメータ数 |
| バッチサイズ `B` | `D^0.383`（Power Lines 論文） |
| 学習率 `η` | `η_ref × √(B / B_ref)` |
| 重み減衰 `λ` | `λ_ref × √(B / B_ref) × (D_ref / D)` |

#### 学習率スケジュール

```
η
│   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
│  ╱   cosine decay          ╲
│ ╱                            ╲___  cooldown
│╱                                 ╲
└──────────────────────────────────── step
 warmup                        cooldown
```

#### 学習ループ

- 勾配累積でマイクロバッチ処理
- `torch.compile` でモデルの forward を高速化
- FP8 学習時は `Float8Linear` のスケールを動的に調整（評価時は `disable_fp8()` で無効化）
- 定期的に BPB（bits-per-byte）と CORE スコア（ARC, MMLU 等の総合指標）で進捗を評価

</details>

<details>
<summary><b><code>chat_sft.py</code> — 教師あり微調整 (SFT)</b></summary>

事前学習済みチェックポイントを読み込み、タスク混合データで微調整する。

#### データ混合

`TaskMixture` で複数タスクを混合:

- SmolTalk（汎用会話）
- MMLU（多肢選択）
- GSM8K（数学）
- SpellingBee

MMLU と GSM8K のエポック数は CLI 引数で調整可能。

#### パッキングとマスク

`sft_data_generator_bos_bestfit()` が会話をトークン化し、ベストフィットで行に詰める。

```
入力: <|bos|> <|user_start|> 質問 <|user_end|> <|assistant_start|> 回答 <|assistant_end|> [PAD]
ロス:  無視     無視          無視    無視          無視              学習     無視      無視
```

アシスタント発話のみをロス計算の対象とし、それ以外は `ignore_index = -1` でマスクする。

#### オプティマイザの温スタート

事前学習のモメンタムバッファを引き継ぎ、学習率のみリセット。
LR スケジュールは「進捗率 0→1」で制御され、データセットサイズが変わっても自動で適応する。

</details>

<details>
<summary><b><code>chat_rl.py</code> — 強化学習 (GRPO)</b></summary>

GSM8K の数学文章題を使い、簡略化された GRPO（Group Relative Policy Optimization）で方策を改善する。

#### なぜ簡略化できるか

**オンポリシー**（毎ステップ最新のモデルでサンプリング）なので、
PPO の比率クリッピングや KL 正則化は不要。

#### 1ステップの流れ

```
① プロンプトから num_samples 個のサンプルを生成
② 各サンプルの回答をデコードし、reward() で正誤判定（0 or 1）
③ アドバンテージ = reward - mean(rewards)  ← 単純な平均ベースライン
④ トークンレベルで正規化した方策勾配を計算
⑤ パラメータ更新
```

#### 評価

pass@k（k 個のサンプル中 1 つ以上正解なら成功）を GSM8K テストセットで計測。
分散環境では `all_reduce` で全ランクの結果を集約する。

</details>

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

<details>
<summary><b><code>common.py</code> — タスク基盤</b></summary>

すべてのベンチマークの基底クラスと合成ユーティリティ。

#### `Task` 基底クラス

サブクラスは `_items` リストを構築するだけで、以下のインターフェースが使える:

```python
task = GSM8K(split="train")
task[0]          # → 会話形式の1サンプル（messages 配列）
task.evaluate()  # → モデル出力の正誤を判定
task.reward()    # → RL 用のスカラー報酬（float）
```

`start`, `stop`, `step` によるスライスで、データの部分集合を柔軟に切り出せる。

#### 合成クラス

| クラス | 用途 |
|---|---|
| `TaskMixture` | 複数タスクを結合し、決定論的にシャッフル（SFT のデータ混合に使用） |
| `TaskSequence` | タスクを順番に連結（カリキュラム学習用） |

</details>

<details>
<summary><b><code>gsm8k.py</code> — GSM8K 数学文章題</b></summary>

GSM8K データセットをラップし、ツール呼び出し形式での解答生成と自動評価を提供する。

#### データの変換

元の解答にある `<<expression>>` パターンを特殊トークンに変換:

```
元:  The cost is <<5*3=15>>15 dollars.
変換: The cost is <|python_start|>5*3<|python_end|><|output_start|>15<|output_end|>15 dollars.
```

#### 評価の仕組み

`evaluate()` はモデル出力から `####` マーカー以降の数値を抽出し、
カンマを除去・正規化した上で正解と比較する。

`reward()` は正解なら `1.0`、不正解なら `0.0` を返す（RL で使用）。

</details>

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
