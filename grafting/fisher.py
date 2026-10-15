"""Estimate the diagonal retention Fisher of a post-trained model.

The mask in grafting/mask.py ranks coordinates by F_ret(j) * delta_j ** 2, where F_ret is the
diagonal Fisher of the post-trained model over a retention set. This script produces F_ret.

    python -m grafting.fisher \
        --model Qwen/Qwen3-1.7B \
        --data retention_rollouts.parquet \
        --out out/fisher_ret

The retention set is prompts the model should keep answering well, paired with responses
SAMPLED FROM THE MODEL ITSELF (temperature 1.0). That is what makes this the Fisher rather
than the empirical Fisher: the expectation is over the model's own distribution, so the score
identity holds and the estimate does not measure how badly the model fits somebody else's
text. Reusing a fixed corpus of human or teacher text here gives a different quantity.

Input is a parquet or jsonl file with a `question` and a `response` column. One estimate is
enough for a given post-trained model; it does not depend on the delta, so it is computed once
and reused across every graft built on that model.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_rows(path):
    if path.endswith(".jsonl"):
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
    import pandas as pd
    return pd.read_parquet(path).to_dict("records")


def build_ids(tok, question, response, max_len, device, chat_template_kwargs):
    """Tokenize prompt plus response and return the ids and where the response starts."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False, **chat_template_kwargs)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None and (not r_ids or r_ids[-1] != tok.eos_token_id):
        r_ids = r_ids + [tok.eos_token_id]
    ids = (p_ids + r_ids)[:max_len]
    resp_start = min(len(p_ids), len(ids))
    if resp_start >= len(ids):
        return None, None
    return torch.tensor([ids], device=device), resp_start


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the post-trained model")
    ap.add_argument("--data", required=True, help="parquet or jsonl with question, response")
    ap.add_argument("--out", required=True, help="output prefix, no extension")
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # MEMORY. In fp32 this needs three copies of the model: weights, gradients, and the
    # accumulator. At 8B on one 80GB card that is 96GB before activations and it does not fit.
    # bfloat16 halves the first two. THE ACCUMULATOR STAYS fp32 EITHER WAY, because it sums
    # squared gradients over thousands of sequences and that is exactly where a low-precision
    # accumulator silently drops the small contributions the ranking depends on.
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="model and gradient dtype; the accumulator is always fp32")
    ap.add_argument("--enable_thinking", default=None,
                    help="passed to apply_chat_template for models that take it")
    args = ap.parse_args()

    rows = _load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    ct_kwargs = {}
    if args.enable_thinking is not None:
        ct_kwargs["enable_thinking"] = args.enable_thinking == "True"
    print(f"{len(rows)} sequences, max_len={args.max_len}, dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=True,
        attn_implementation="eager" if args.device == "cpu" else "sdpa")
    model.to(args.device)
    model.gradient_checkpointing_enable()
    model.train()  # gradients are required; dropout is 0 in these configs

    fisher = {n: torch.zeros_like(p, dtype=torch.float32)
              for n, p in model.named_parameters() if p.requires_grad}
    total_tokens = n_seqs = 0

    for row in rows:
        ids, resp_start = build_ids(tok, row["question"], row["response"],
                                    args.max_len, args.device, ct_kwargs)
        if ids is None:
            continue
        model.zero_grad(set_to_none=True)
        logits = model(ids).logits[0].float()
        # position t predicts token t+1, so the response is predicted from resp_start-1 onward
        pred = logits[resp_start - 1: -1]
        tgt = ids[0, resp_start:]
        F.cross_entropy(pred, tgt, reduction="sum").backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.float() ** 2
        total_tokens += int(tgt.numel())
        n_seqs += 1
        if n_seqs % 25 == 0:
            print(f"  {n_seqs} sequences, {total_tokens} tokens", flush=True)

    # Per-token normalization. Only the relative ordering of coordinates enters the mask, so the
    # constant does not matter on its own, but it must be the same one everywhere the Fisher is
    # compared or combined.
    for n in fisher:
        fisher[n] /= max(total_tokens, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from safetensors.torch import save_file
    save_file({n: t.cpu().contiguous() for n, t in fisher.items()}, args.out + ".safetensors")
    json.dump({"model": args.model, "data": args.data, "n_seqs": n_seqs,
               "total_tokens": total_tokens, "max_len": args.max_len,
               "normalization": "per_token"},
              open(args.out + ".meta.json", "w"), indent=1)
    print(f"wrote {args.out}.safetensors ({n_seqs} sequences, {total_tokens} tokens)")


if __name__ == "__main__":
    main()
