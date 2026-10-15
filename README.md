# Grafting

Reference implementation for the paper:<br>
**[Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning](https://arxiv.org/abs/2610.05872)**<br>
Chen Henry Wu, Thomas Zhang, Aditi Raghunathan<br>
Preprint 2026

Project page: https://ar-forum.github.io/grafting-webpage/

**Grafting** adds a skill or new knowledge to a post-trained model: finetune an earlier checkpoint
(the *donor*) on the new data, then add the scaled, masked weight update to the post-trained model.

```
theta(lam, rho) = theta_post + lam * ( m_rho * (theta_donor_ft - theta_donor) )
```

| symbol | meaning |
| --- | --- |
| `theta_post` | the post-trained model |
| `theta_donor` | the donor, e.g. the base model (`--base` in the code) |
| `theta_donor_ft` | the donor finetuned on the new data (`--base_ft`) |
| `lam` | scale of the update, between 0 and 1 |
| `m_rho` | Fisher mask that drops the `rho` fraction of coordinates most costly to `theta_post` |

## Install

```bash
pip install -r requirements.txt
pip install -e third_party/verl                # for training only
pip install flash-attn --no-build-isolation    # optional
pip install vllm datasets                      # for evaluation only
```

## Data

[`ChenWu98/grafting-data`](https://huggingface.co/datasets/ChenWu98/grafting-data) on HuggingFace:

| config | splits | contents |
| --- | --- | --- |
| `math_qwen3_1p7b_thinking` | train 1,533 / heldout 111 | teacher traces for Qwen3-1.7B (thinking) |
| `math_qwen3_1p7b_thinking_star` | train 1,093 | STaR traces from Qwen3-1.7B (thinking) |
| `math_qwen3_8b_thinking` | train 985 / heldout 100 | teacher traces for Qwen3-8B (thinking) |
| `math_qwen3_8b_nonthinking` | train 1,163 / heldout 200 | worked solutions for Qwen3-8B (non-thinking) |
| `math_olmo3_7b_thinking` | train 730 / heldout 100 | teacher traces for OLMo-3-7B (thinking) |
| `knowledge` | train 9,000 / train_mix 10,533 / heldout 2,250 | QA on fictional entities, the mix with math replay, held-out paraphrases |
| `knowledge_articles` | 200 | articles stating the facts |
| `retention_qwen3_1p7b` | 1,858 | Qwen3-1.7B rollouts, for the Fisher |
| `old_task` | aime24 / aime2025 / hmmt2025, 30 each | old-task benchmarks |

## Train

`scripts/train_sft.sh` runs the paper's SFT recipe with the verl trainer in `third_party/verl`
([our changes](third_party/verl/README.md)). The same script trains the donor and the direct SFT
baseline (`MODEL` = the post-trained model).

```bash
python scripts/prepare_sft_data.py --config math_qwen3_1p7b_thinking --out data/math_1p7b.parquet
MODEL=Qwen/Qwen3-1.7B-Base TRAIN_FILE=data/math_1p7b.parquet LR=1e-5 \
    SAVE_DIR=ckpt/math_base_ft bash scripts/train_sft.sh

python scripts/prepare_sft_data.py --config knowledge --split train_mix --out data/knowledge_mix.parquet
MODEL=Qwen/Qwen3-1.7B-Base TRAIN_FILE=data/knowledge_mix.parquet LR=2e-5 \
    SAVE_DIR=ckpt/knowledge_base_ft bash scripts/train_sft.sh
```

Each epoch is saved as a HuggingFace model at `$SAVE_DIR/global_step_<step>/huggingface`.

| setting | value |
| --- | --- |
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01, grad clip 1.0 |
| schedule | cosine, warmup ratio 0.1 |
| batch / length | 32 / 18,432 tokens |
| epochs | 5, checkpoint each epoch |
| learning rate | math: swept 5e-7 to 5e-5; knowledge: 2e-5 |

## Use

### 1. Graft

```bash
python -m grafting.graft \
    --base    Qwen/Qwen3-1.7B-Base \
    --base_ft ckpt/math_base_ft/global_step_235/huggingface \
    --post    Qwen/Qwen3-1.7B \
    --lam 0.5 --out out/graft_lam0.5
```

Sweep `--lam` to trace the trade-off between the new skill and the old ones.

### 2. Fisher mask (optional)

```bash
python -m grafting.fisher --model Qwen/Qwen3-1.7B --data retention_rollouts.parquet --out out/fisher_ret

python -m grafting.graft \
    --base    Qwen/Qwen3-1.7B-Base \
    --base_ft ckpt/math_base_ft/global_step_235/huggingface \
    --post    Qwen/Qwen3-1.7B \
    --fisher  out/fisher_ret.safetensors \
    --lam 0.5 --protect 0.01 --out out/graft_lam0.5_rho0.01
```

`--data` needs `question` and `response` columns, with responses sampled from the post-trained model
itself at temperature 1.0 (e.g. `retention_qwen3_1p7b`). `--protect 0.01` drops the 1% of coordinates
with the highest `F(j) * delta_j^2`. Sweep `lam` and `rho` together.

### 3. Other donors

Any earlier checkpoint can be the donor; `--base_ft` must be that same checkpoint after finetuning.

```bash
# a post-training stage (OLMo-3 SFT checkpoint)
python -m grafting.graft --base allenai/Olmo-3-7B-Think-SFT --base_ft ckpt/sft_ft \
    --post allenai/Olmo-3-7B-Think --lam 0.5 --out out/graft_from_sft

# a pretraining checkpoint (a revision of the base repo)
python -m grafting.graft --base allenai/Olmo-3-1025-7B --base_revision stage3-step6000 \
    --base_ft ckpt/step6000_ft --post allenai/Olmo-3-7B-Think --lam 0.5 --out out/graft_from_step6000

# the post-trained model itself = SFT followed by merging (baseline)
python -m grafting.graft --base allenai/Olmo-3-7B-Think --base_ft ckpt/post_ft \
    --post allenai/Olmo-3-7B-Think --lam 0.5 --out out/sft_merge_lam0.5
```

`allenai/Olmo-3-1025-7B` has no chat template; copy it from `allenai/Olmo-3-7B-Think` before training.

## Evaluate

```bash
python -m grafting.evaluate --model out/graft_lam0.5 --task math --thinking \
    --config math_qwen3_1p7b_thinking --split heldout --out out/eval_heldout.json
python -m grafting.evaluate --model out/graft_lam0.5 --task math --thinking \
    --config old_task --split aime24 --out out/eval_aime24.json
python -m grafting.evaluate --model out/graft_lam0.75 --task knowledge \
    --config knowledge --split heldout --out out/eval_knowledge.json
```

| task | samples | decoding | max new tokens | score |
| --- | --- | --- | --- | --- |
| math, thinking | 16 | temperature 1.0, top-p 1.0 | 32,768 | `math_score` (math-verify on the full generation) |
| math, non-thinking | 16 | temperature 1.0, top-p 1.0 | 16,384 | same |
| knowledge | 1 | greedy, non-thinking | 192 | `knowledge_score` (exact match inside `<answer>`) |

Old-task performance is the mean over AIME 2024, AIME 2025 and HMMT 2025.

## Citation

```bibtex
@article{wu2026grafting,
  title   = {Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning},
  author  = {Wu, Chen Henry and Zhang, Thomas and Raghunathan, Aditi},
  journal = {arXiv preprint arXiv:2610.05872},
  year    = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE).
