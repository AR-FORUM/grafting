"""Evaluate a model with the paper's settings.

Math (held-out problems, AIME 2024, AIME 2025, HMMT 2025): 16 samples per problem at temperature 1.0
and top-p 1.0, each scored by `grafting.math_score`; we report the mean score (pass@1). Thinking mode
generates up to 32,768 tokens and non-thinking mode up to 16,384.

Knowledge (held-out paraphrases): one greedy sample per question in non-thinking mode, up to 192
tokens, scored by exact match of the <answer> text (`grafting.knowledge_score`).

    # math, thinking mode, on the held-out problems
    python -m grafting.evaluate --model out/graft_lam0.5 --task math --thinking \
        --config math_qwen3_1p7b_thinking --split heldout --out out/eval_heldout.json

    # math, the old-task benchmarks
    python -m grafting.evaluate --model out/graft_lam0.5 --task math --thinking \
        --config old_task --split aime24 --out out/eval_aime24.json

    # knowledge
    python -m grafting.evaluate --model out/graft_lam0.75 --task knowledge \
        --config knowledge --split heldout --out out/eval_knowledge.json

Data comes from the `--config` / `--split` of the HuggingFace dataset `--dataset`, or from a local
parquet or jsonl file passed as `--data`. Either way each row needs a `question` (the full user message)
and an `answer`. Needs vllm and datasets on top of requirements.txt.
"""
import argparse
import json
import os

import pandas as pd

MATH = dict(n=16, temperature=1.0, top_p=1.0)
KNOWLEDGE = dict(n=1, temperature=0.0, top_p=1.0, max_tokens=192, max_model_len=2048)


def _load(args):
    if args.data:
        return pd.read_json(args.data, lines=True) if args.data.endswith(".jsonl") \
            else pd.read_parquet(args.data)
    from datasets import load_dataset
    return load_dataset(args.dataset, args.config, split=args.split).to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["math", "knowledge"])
    ap.add_argument("--dataset", default="ChenWu98/grafting-data")
    ap.add_argument("--config", help="dataset config, e.g. math_qwen3_1p7b_thinking or old_task")
    ap.add_argument("--split", help="dataset split, e.g. heldout or aime24")
    ap.add_argument("--data", help="local parquet or jsonl instead of --dataset/--config/--split")
    ap.add_argument("--out", required=True, help="output json with the score and every generation")
    ap.add_argument("--thinking", action="store_true",
                    help="math only: Qwen3 thinking mode (knowledge is always non-thinking)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.data and not (args.config and args.split):
        ap.error("pass --data, or both --config and --split")

    if args.task == "math":
        from .math_score import compute_score
        cfg = dict(MATH)
        cfg["max_tokens"], cfg["max_model_len"] = (32768, 36864) if args.thinking else (16384, 20480)
        thinking = args.thinking
        score = lambda text, gt: compute_score("math", text, gt)  # noqa: E731
    else:
        from .knowledge_score import compute_score as score
        cfg = dict(KNOWLEDGE)
        thinking = False

    from vllm import LLM, SamplingParams
    df = _load(args)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=cfg["max_model_len"], seed=args.seed)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=thinking)
               for q in df["question"]]
    sp = SamplingParams(n=cfg["n"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                        max_tokens=cfg["max_tokens"], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rows = []
    for q, gt, o in zip(df["question"], df["answer"], outs):
        scores = [score(c.text, str(gt)) for c in o.outputs]
        rows.append({"question": q, "answer": str(gt), "score": sum(scores) / len(scores),
                     "scores": scores, "generations": [c.text for c in o.outputs]})
    mean = sum(r["score"] for r in rows) / len(rows)
    summary = {"model": args.model, "task": args.task, "thinking": thinking, "n_problems": len(rows),
               **cfg, "score": mean}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
