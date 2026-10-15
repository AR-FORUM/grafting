#!/bin/bash
# Supervised finetuning with the paper's settings, using the verl FSDP SFT trainer in third_party/verl.
# The same recipe trains the donor (MODEL = the donor, e.g. the base model) and the direct SFT baseline
# (MODEL = the post-trained model).
#
#   python scripts/prepare_sft_data.py --config math_qwen3_1p7b_thinking --out data/math_1p7b.parquet
#   MODEL=Qwen/Qwen3-1.7B-Base TRAIN_FILE=data/math_1p7b.parquet LR=1e-5 \
#       SAVE_DIR=ckpt/base_ft_lr1e-5 bash scripts/train_sft.sh
#
# Settings (paper, training appendix): AdamW with betas (0.9, 0.95) and weight decay 0.01, cosine schedule
# with warmup ratio 0.1, gradient clipping 1.0 (verl's defaults), global batch 32, sequence length 18,432,
# 5 epochs with a checkpoint after each. The learning rate is the knob swept in the paper; knowledge uses
# 2e-5. Each checkpoint is converted to a HuggingFace model at
# $SAVE_DIR/global_step_<step>/huggingface, which is what grafting.graft takes as --base_ft.
#
# Run from the repository root. Any extra arguments are passed to verl as config overrides.
set -euo pipefail

MODEL=${MODEL:?set MODEL, e.g. Qwen/Qwen3-1.7B-Base}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE to a parquet written by scripts/prepare_sft_data.py}
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR, e.g. ckpt/base_ft}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
NPROC=${NPROC:-8}               # GPUs on this node
BATCH_SIZE=${BATCH_SIZE:-32}    # global batch
MICRO_BSZ=${MICRO_BSZ:-1}       # per-GPU micro-batch; the global batch is reached by gradient accumulation
MAX_LENGTH=${MAX_LENGTH:-18432}
MASTER_PORT=${MASTER_PORT:-29500}
MERGE=${MERGE:-1}               # 1: convert every checkpoint to a HuggingFace model after training

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Remove-padding needs the flash-attn varlen kernels; without flash-attn, train padded with SDPA.
USE_RMPAD=$(python -c "import importlib.util; print(str(importlib.util.find_spec('flash_attn') is not None).lower())")

# One checkpoint per epoch. Steps per epoch as verl counts them: each rank drops the remainder of
# len/NPROC, then the remainder of its batches.
ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('${TRAIN_FILE}')))")
SAVE_FREQ=$(( (ROWS / NPROC) / (BATCH_SIZE / NPROC) ))
echo "rows=${ROWS} steps/epoch=${SAVE_FREQ} epochs=${TOTAL_EPOCHS} lr=${LR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.val_max_samples=64 \
    data.prompt_key=question \
    data.response_key=response \
    data.train_batch_size="${BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    data.max_length="${MAX_LENGTH}" \
    optim.lr="${LR}" \
    model.partial_pretrain="${MODEL}" \
    model.strategy=fsdp \
    model.trust_remote_code=True \
    use_remove_padding="${USE_RMPAD}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name=grafting \
    trainer.experiment_name="$(basename "${SAVE_DIR}")" \
    trainer.logger='["console"]' \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    'trainer.checkpoint.save_contents=[model,extra]' \
    "$@"

if [ "${MERGE}" = "1" ]; then
    for CK in "${SAVE_DIR}"/global_step_*; do
        [ -d "${CK}/huggingface" ] && ls "${CK}"/huggingface/model*.safetensors >/dev/null 2>&1 && continue
        CUDA_VISIBLE_DEVICES="" python -m verl.model_merger merge --backend fsdp \
            --local_dir "${CK}" --target_dir "${CK}/huggingface"
    done
fi
