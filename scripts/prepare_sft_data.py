"""Write one training split of the data as the parquet file scripts/train_sft.sh reads.

    python scripts/prepare_sft_data.py --config math_qwen3_1p7b_thinking --out data/math_1p7b.parquet
    python scripts/prepare_sft_data.py --config knowledge --split train_mix --out data/knowledge_mix.parquet

The file has `question` (the full user message), `response` (the target) and `enable_thinking`. The
last one is the chat-template mode each row is rendered in: true when the response starts with a
`<think>` block, false otherwise. In false mode the Qwen3 template puts the empty think block in the
prompt, so it is never a training target. This is how the knowledge mix trains its answer-only QA rows
and its math replay rows (full reasoning) in one run.
"""
import argparse
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ChenWu98/grafting-data")
    ap.add_argument("--config", required=True, help="e.g. math_qwen3_1p7b_thinking or knowledge")
    ap.add_argument("--split", default="train", help="e.g. train, or train_mix for the knowledge mix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_dataset(args.dataset, args.config, split=args.split).to_pandas()
    df = df[["question", "response"]].copy()
    df["enable_thinking"] = df["response"].str.startswith("<think>")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {int(df['enable_thinking'].sum())} in thinking mode")


if __name__ == "__main__":
    main()
