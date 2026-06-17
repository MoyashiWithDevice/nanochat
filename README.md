# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

<details>
<summary>🇯🇵 日本語版 / Japanese</summary>

## nanochat

nanochatは、LLMの学習のための最もシンプルな実験用ハーネスです。単一GPUノードで動作するよう設計されており、コードは最小限でハックしやすく、トークナイゼーション、事前学習、ファインチューニング、評価、推論、チャットUIなど、LLMの主要なステージをすべてカバーしています。例えば、2019年に約$43,000かかったGPT-2相当のLLMを、わずか$48（8×H100 GPUノードで約2時間）で学習し、ChatGPTライクなWebUIで対話できます。スポットインスタンスを使えば、総コストは約$15まで下げられます。より一般的には、nanochatはGPTトランスフォーマーモデルのレイヤー数である`--depth`という1つの複雑性ダイヤルを設定するだけで、計算最適なモデルのミニシリーズ全体を学習できるよう設定されています（GPT-2相当の性能はおよそdepth 26に相当します）。その他すべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、学習ホライズン、重み減衰など）は最適な形で自動計算されます。

リポジトリに関する質問は、Devin/Cognitionの[DeepWiki](https://deepwiki.com/MoyashiWithDevice/nanochat)でリポジトリについて質問するか、[Discussionsタブ](https://github.com/MoyashiWithDevice/nanochat/discussions)を使用するか、Discordの[#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031)チャンネルをご利用ください。

## Time-to-GPT-2 リーダーボード

現在の開発の主な焦点は、最も計算量を要する事前学習ステージのチューニングです。modded-nanogptリポジトリに触発され、進歩とコミュニティの協力を促進するために、nanochatは「GPT-2スピードラン」のリーダーボードを維持しています。これは、DCLM COREスコアで測定されるGPT-2相当の性能に到達するまでの実時間です。[runs/speedrun.sh](runs/speedrun.sh)スクリプトは、GPT-2相当のモデルを学習し対話するための参照方法を常に反映しています。

| # | 時間 | val_bpb | CORE | 説明 | 日付 | コミット | 貢献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168時間 | - | 0.2565 | オリジナルOpenAI GPT-2チェックポイント | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24ベースライン、やや過学習 | 2026年1月29日 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26やや学習不足 **+fp8** | 2026年2月2日 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 総バッチサイズを1Mトークンに増加 | 2026年2月5日 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | データセットをNVIDIA ClimbMixに変更 | 2026年3月4日 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | 自動研究 [ラウンド1](https://x.com/karpathy/status/2031135152349524125) | 2026年3月9日 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | 自動研究 ラウンド2 | 2026年3月14日 | a825e63 | @karpathy |

主要な指標は「Time to GPT-2」- 8×H100 GPUノードでGPT-2（1.6B）のCORE指標を上回るのに必要な実時間です。GPT-2のCOREスコアは0.256525です。2019年にはGPT-2の学習に約$43,000かかりましたが、7年間にわたるスタック全体の多くの進歩により、現在ではるかに高速に$100未満で達成できます（例：現在の約$3/GPU/hrでは、8×H100ノードは約$24/hrなので、2時間で約$48）。

リーダーボードの解釈と貢献方法の詳細は[dev/LEADERBOARD.md](dev/LEADERBOARD.md)を参照してください。

## はじめに

### セットアップ

nanochatは依存関係管理に[uv](https://docs.astral.sh/uv/)を使用しています。インストール方法：

```bash
uv sync --extra gpu    # CUDA用（A100/H100等）
uv sync --extra cpu    # （または）CPU専用 / MPS用
source .venv/bin/activate
```

開発用（pytest、matplotlib、ipykernel、transformersなどを追加）：

```bash
uv sync --extra gpu --group dev
```

### GPT-2の再現と対話

最も楽しい体験は、自分自身のGPT-2を学習して対話することです。そのための全パイプラインは単一ファイル[runs/speedrun.sh](runs/speedrun.sh)に含まれており、8×H100 GPUノードで実行するよう設計されています。お好みのプロバイダー（例：[Lambda](https://lambda.ai/service/gpu-cloud)）から新しい8×H100 GPUボックスを起動し、学習スクリプトを実行します：

```bash
bash runs/speedrun.sh
```

これは約3時間かかるため、screenセッションで実行することをお勧めします。完了後、ChatGPTライクなWebUIで対話できます。ローカルのuv仮想環境が有効であることを確認し（`source .venv/bin/activate`を実行）、サーブします：

```bash
python -m scripts.chat_web
```

表示されたURLにアクセスしてください。例えばLambdaでは、ノードのパブリックIPにポートを続けてアクセスします（例：[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)）。そして、ChatGPTと普段通りに対話してください！物語や詩を書かせたり、自分が誰かを聞いてハルシネーションを見たり、空がなぜ青いか（または緑か）を聞いたりしてみましょう。スピードランは4e19 FLOPsの性能モデルなので、幼稚園児と話すような感覚です :)

---

いくつかの追加メモ：

- コードはAmpere 8×A100 GPUノードでも問題なく動作しますが、やや遅くなります。
- `torchrun`を省略すれば単一GPUでも動作し、ほぼ同一の結果が得られます（コードは自動的に勾配累積に切り替わります）が、8倍の時間がかかります。
- GPU(s)のVRAMが80GB未満の場合、ハイパーパラメータの調整が必要です。スクリプト内の`--device-batch-size`を探し、収まるまで減らしてください（例：32（デフォルト）→16、8、4、2、または1）。
- コードの大部分は標準的なPyTorchなので、xpu、mpsなどサポートするもので動作するはずですが、すべてのコードパスは個人的に検証していないため、エッジケースがあるかもしれません。

## 研究

nanochatの改善に貢献したい研究者の方には、[runs/scaling_laws.sh](runs/scaling_laws.sh)と[runs/miniseries.sh](runs/miniseries.sh)の2つのスクリプトが参考になります。関連ドキュメントは[Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)を参照してください。素早い実験（約5分の事前学習）には、12層モデル（GPT-1サイズ）の学習がお気に入りです：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これはwandb（実行名"d12"）を使用し、COREメトリックは最終ステップのみで実行され、中間チェックポイントのサンプリングと保存は行いません。コードを変更し、d12（またはd16など）を再実行して改善されたかを確認する反復ループが好みです。実行が改善されたかを確認するには、wandbプロットを監視します：

1. `val_bpb`（語彙サイズに依存しないビット/バイト単位の検証損失）を`step`、`total_training_time`、`total_training_flops`の関数として。
2. `core_metric`（DCLM COREスコア）
3. VRAM使用率、`train/mfu`（モデルFLOPS使用率）、`train/tok_per_sec`（学習スループット）

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照。

重要な点として、nanochatはトランスフォーマーのdepthという1つの複雑性ダイヤルを中心に記述・設定されています。この単一の整数が他のすべてのハイパーパラメータを自動的に決定し、学習されたモデルが計算最適になります。ユーザーはこれらについて考えたり設定したりする必要はなく、単に`--depth`を使ってより小さいまたはより大きいモデルを要求するだけで、すべてが「うまく動く」ようになっています。depthをスイープすることで、さまざまなサイズの計算最適モデルのnanochatミニシリーズが得られます。GPT-2性能モデル（現時点で最も関心が高い）は、現在のコードでd24-d26の範囲にあります。ただし、リポジトリへの候補変更は、depthのすべての設定で機能する十分に原理的なものでなければなりません。

## CPU / MPSでの実行

スクリプト[runs/runcpu.sh](runs/runcpu.sh)は、CPUまたはApple Siliconで実行するための非常にシンプルな例を示しています。学習するLLMを劇的に小さくして、数十分の学習で収まるようにしています。この方法では強い結果は得られません。

## 精度 / dtype

nanochatは`torch.amp.autocast`を使用しません。代わりに、精度は単一のグローバル`COMPUTE_DTYPE`（`nanochat/common.py`で定義）を通じて明示的に管理されます。デフォルトではハードウェアに基づいて自動検出されます：

| ハードウェア | デフォルトdtype | 理由 |
|----------|--------------|-----|
| CUDA SM 80+（A100、H100、...） | `bfloat16` | ネイティブbf16テンソルコア |
| CUDA SM < 80（V100、T4、...） | `float32` | bf16なし; fp16は`NANOCHAT_DTYPE=float16`で利用可能（GradScaler使用） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

`NANOCHAT_DTYPE`環境変数でデフォルトを上書きできます：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # fp32を強制
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # bf16を強制
```

仕組み：モデルの重みはfp32で保存されます（オプティマイザの精度のため）が、カスタム`Linear`レイヤーがフォワードパス中に`COMPUTE_DTYPE`にキャストします。エンベディングはメモリ節約のために直接`COMPUTE_DTYPE`で保存されます。これにより、どの精度で何が実行されるかを完全に明示的に制御しながら、autocastと同じ混合精度の利点が得られます。

注意：`float16`学習は`base_train.py`で`GradScaler`を自動的に有効にして勾配アンダーフローを防止します。SFTもこれをサポートしていますが、RLは現在サポートしていません。fp16での推論はどこでも問題なく動作します。

## ガイド

役立つ情報を含むいくつかのガイドを公開しています（新しい順）：

- [2026年2月1日: $100未満でGPT-2を超える：nanochatの旅](https://github.com/karpathy/nanochat/discussions/481)
- [1月7日 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) - 最初のnanochatモデルミニシリーズを文書化。
- nanochatに新しい能力を追加するには、[ガイド：strawberryのrを数える（および一般的に能力を追加する方法）](https://github.com/karpathy/nanochat/discussions/164)を参照。
- nanochatをカスタマイズするには、Discussionsの[ガイド：nanochatにアイデンティティを注入する](https://github.com/karpathy/nanochat/discussions/139)を参照。合成データ生成とSFTステージへのデータ混合によるパーソナリティの調整方法を説明しています。
- [2025年10月13日: nanochat紹介ポスト](https://github.com/karpathy/nanochat/discussions/1) - nanochatを紹介していますが、現在は一部非推奨の情報を含み、モデルは現在のmasterよりもはるかに古い（結果も劣る）ものです。

## ファイル構造

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用合成データ例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データシャード生成
├── nanochat
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み
│   ├── common.py                   # 各種小ユーティリティ
│   ├── core_eval.py                # ベースモデルCOREスコア評価（DCLM論文）
│   ├── dataloader.py               # トークン化分散データローダー
│   ├── dataset.py                  # 事前学習データのダウンロード/読み取りユーティリティ
│   ├── engine.py                   # KVキャッシュ付き効率的モデル推論
│   ├── execution.py                # LLMによるPythonコード実行（ツール）
│   ├── flash_attention.py          # Flash Attention / SDPAラッパー
│   ├── fp8.py                      # H100学習用FP8 Linearレイヤー
│   ├── gpt.py                      # GPT nn.Moduleトランスフォーマー
│   ├── logo.svg
│   ├── loss_eval.py                # ビット/バイト評価（損失の代わりに）
│   ├── optim.py                    # AdamW + Muonオプティマイザ、1GPUおよび分散
│   ├── report.py                   # nanochatレポート書き込みユーティリティ
│   ├── tokenizer.py                # GPT-4スタイルのBPEトークナイザラッパー
│   └── ui.html                     # nanochatフロントエンドHTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズ学習スクリプト
│   ├── runcpu.sh                   # CPU/MPSでの実行例
│   ├── scaling_laws.sh             # スケーリング則実験
│   └── speedrun.sh                 # ~$100 nanochat d20の学習
├── scripts
│   ├── base_eval.py                # ベースモデル：COREスコア、BPB、サンプル
│   ├── base_train.py               # ベースモデル：学習
│   ├── chat_cli.py                 # チャットモデル：CLI対話
│   ├── chat_eval.py                # チャットモデル：タスク評価
│   ├── chat_rl.py                  # チャットモデル：強化学習
│   ├── chat_sft.py                 # チャットモデル：SFT学習
│   ├── chat_web.py                 # チャットモデル：WebUI対話
│   ├── tok_eval.py                 # トークナイザ：圧縮率評価
│   └── tok_train.py                # トークナイザ：学習
├── tasks
│   ├── arc.py                      # 多肢選択科学問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意のjsonl会話からTask作成
│   ├── gsm8k.py                    # 8K小学校算数問題
│   ├── humaneval.py                # 誤称; シンプルPythonコーディングタスク
│   ├── mmlu.py                     # 多肢選択問題、幅広いトピック
│   ├── smoltalk.py                 # HFのSmolTalk統合データセット
│   └── spellingbee.py              # スペル/文字数え教育タスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## 貢献

nanochatの目標は、$1000未満の予算でエンドツーエンドに扱えるマイクロモデルの最先端を改善することです。アクセシビリティはコストだけでなく認知的複雑性にも関係します - nanochatは網羅的に設定可能なLLM「フレームワーク」ではありません。巨大な設定オブジェクト、モデルファクトリー、if-then-elseの怪物はコードベースにありません。最初から最後まで実行して対話可能なChatGPTモデルを生成する、単一の、一貫性のある、最小限の、読みやすい、ハックしやすい、最大限フォーク可能な「強力なベースライン」コードベースです。現在、個人的に最も興味深い部分は、GPT-2までのレイテンシを高速化すること（つまり、COREスコア0.256525以上を達成すること）です。現在これには約3時間かかりますが、事前学習ステージを改善することでさらに短縮できます。

現在のAIポリシー：開示。PRを提出する際、LLMが実質的に貢献した部分で、自分で書いていない、または完全に理解していない部分を宣言してください。

## 謝辞

- 名前（nanochat）は、事前学習のみをカバーしていた以前のプロジェクト[nanoGPT](https://github.com/karpathy/nanoGPT)に由来しています。
- nanochatは[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)にもインスパイアされています。nanoGPTリポジトリを明確なメトリクスとリーダーボードでゲーミフィケーションし、事前学習のアイデアと実装の多くを借用しています。
- finwebとsmoltalkについて[HuggingFace](https://huggingface.co/)に感謝します。
- このプロジェクトの開発に使用された計算リソースについて[Lambda](https://lambda.ai/service/gpu-cloud)に感謝します。
- アドバイス/ガイダンスについてチーフLLMウィスパラー 🧙‍♂️ Alec Radfordに感謝します。
- nanochatのイシュー、プルリクエスト、ディスカッションの管理を手伝ってくれたリポジトリ管理者のSofie [@svlandeg](https://github.com/svlandeg)に感謝します。

## 引用

nanochatが研究に役立った場合、以下のように引用してください：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT

</details>

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of 8XH100 GPU node) and then talk to it in a familiar ChatGPT-like web UI. On a spot instance, the total cost can be closer to ~$15. More generally, nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability happens to be approximately depth 26). All other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) are calculated automatically in an optimal way.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/MoyashiWithDevice/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/MoyashiWithDevice/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train GPT-2 grade model and talk to it. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 2 hours is ~$48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~3 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── flash_attention.py          # Flash Attention / SDPA wrapper
│   ├── fp8.py                      # FP8 Linear layer for H100 training
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
