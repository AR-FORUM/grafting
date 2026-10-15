"""Build a grafted model.

    theta(lam, rho) = theta_post + lam * ( m_rho * (theta_base_ft - theta_base) )

theta_base is the checkpoint the delta was trained from, theta_base_ft is that checkpoint
after finetuning on the new data, and theta_post is the deployed post-trained model the delta
is written into. m_rho zeroes the top-rho fraction of coordinates by retention damage; with
--protect 0 there is no mask and this is a plain lambda-scaled merge.

    # lambda-scaled task-vector graft, no mask
    python -m grafting.graft \
        --base Qwen/Qwen3-1.7B-Base \
        --base_ft ckpt/base_ft \
        --post Qwen/Qwen3-1.7B \
        --lam 0.5 --out out/graft_lam0.5

    # with the retention mask
    python -m grafting.graft \
        --base Qwen/Qwen3-1.7B-Base \
        --base_ft ckpt/base_ft \
        --post Qwen/Qwen3-1.7B \
        --fisher out/fisher_ret.safetensors \
        --lam 0.5 --protect 0.01 --out out/graft_lam0.5_rho0.01

All three checkpoints must share an architecture and a vocabulary. Arithmetic is done in fp32
and the result is saved in bf16 with the post-trained model's tokenizer and config.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mask import apply_mask, damage_threshold


def _state_dict(path, revision=None):
    model = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, torch_dtype=torch.float32, trust_remote_code=True)
    return model.state_dict()


def _fisher_key(name, fisher):
    """Fisher keys come from named_parameters, which can differ from state_dict by a prefix."""
    if name in fisher:
        return name
    stripped = name.replace("model.", "", 1)
    return stripped if stripped in fisher else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="checkpoint the delta was trained from")
    # Some releases expose their intermediate checkpoints as revisions of one repository rather
    # than as separate repositories, which is how you move theta_base along the training
    # trajectory. Whatever revision you name here must be the one --base_ft was finetuned from;
    # subtracting a different checkpoint still produces a model, and a plausible-looking one.
    ap.add_argument("--base_revision", default=None,
                    help="git revision of --base, for repositories that ship checkpoints as tags")
    ap.add_argument("--base_ft", required=True, help="that checkpoint after finetuning")
    ap.add_argument("--post", required=True, help="post-trained model the delta is written into")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="global scale on the delta")
    ap.add_argument("--protect", type=float, default=0.0,
                    help="fraction of highest-damage coordinates to zero; 0 disables the mask")
    ap.add_argument("--fisher", default=None,
                    help="retention Fisher .safetensors, required when --protect > 0")
    args = ap.parse_args()

    if args.protect and not args.fisher:
        raise SystemExit("--protect > 0 needs --fisher; see grafting/fisher.py")

    fisher = {}
    if args.protect:
        from safetensors.torch import load_file
        fisher = load_file(args.fisher)

    print("loading post-trained model (the graft target)...")
    post = AutoModelForCausalLM.from_pretrained(
        args.post, torch_dtype=torch.float32, trust_remote_code=True)
    psd = post.state_dict()
    print("loading base...")
    bsd = _state_dict(args.base, args.base_revision)
    print("loading base_ft...")
    fsd = _state_dict(args.base_ft)

    deltas, unmatched = {}, 0
    for k in psd:
        if k in bsd and k in fsd:
            deltas[k] = fsd[k] - bsd[k]
        else:
            unmatched += 1
    if not deltas:
        raise SystemExit("no parameter is shared by all three checkpoints; check the models match")

    threshold, fmap = None, {}
    if args.protect:
        fmap = {k: _fisher_key(k, fisher) for k in deltas}
        # A parameter with no Fisher entry cannot be ranked, so it is left unmasked rather than
        # dropped. Buffers and tied weights land here; if the count is large the Fisher does not
        # belong to this model and the mask is meaningless, so we refuse instead of guessing.
        covered = {k: v for k, v in fmap.items() if v is not None}
        if len(covered) < 0.5 * len(deltas):
            raise SystemExit(
                f"the Fisher covers only {len(covered)} of {len(deltas)} parameters; it was "
                "probably estimated on a different model")
        threshold = damage_threshold({k: deltas[k] for k in covered},
                                     {covered[k]: fisher[covered[k]] for k in covered},
                                     args.protect)
        print(f"damage threshold {threshold:.6g} at protect={args.protect}")

    n_protected = n_total = 0
    for k, delta in deltas.items():
        if args.protect and fmap.get(k) is not None:
            delta, n = apply_mask(delta, fisher[fmap[k]], threshold)
            n_protected += n
        psd[k] = psd[k] + args.lam * delta
        n_total += delta.numel()

    realized = n_protected / max(n_total, 1)
    print(f"grafted {len(deltas)} tensors, {unmatched} unmatched, lam={args.lam}, "
          f"protected {realized:.4%} of coordinates")

    post.load_state_dict(psd)
    post.to(torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    post.save_pretrained(args.out)
    # Copying the tokenizer is a convenience, so a checkpoint that ships without one should not
    # kill a build whose weights are already written and correct.
    try:
        AutoTokenizer.from_pretrained(args.post, trust_remote_code=True).save_pretrained(args.out)
    except Exception as exc:
        print(f"warning: could not copy the tokenizer from {args.post} ({exc}); "
              "the weights are saved, copy it yourself before serving the model")
    json.dump({"base": args.base, "base_revision": args.base_revision,
               "base_ft": args.base_ft, "post": args.post,
               "lam": args.lam, "protect": args.protect, "fisher": args.fisher,
               "damage_threshold": threshold, "realized_protected_fraction": realized,
               "n_tensors": len(deltas), "n_unmatched": unmatched},
              open(os.path.join(args.out, "graft_meta.json"), "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
