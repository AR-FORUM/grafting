"""The retention-damage mask: which coordinates of a weight delta to drop.

Grafting writes a delta into a post-trained model. Some coordinates of that delta cost the
model more than others, and the second-order expansion of the retention loss around the
post-trained weights says how much: coordinate j costs about

    s_j = F_ret(j) * delta_j ** 2

where F_ret is the diagonal retention Fisher (see grafting/fisher.py). The mask zeroes the
top-`protect` fraction of coordinates by s_j and leaves the rest untouched.

The threshold is a quantile over every coordinate in the model, which is billions of values,
so it is estimated from a strided subsample rather than by sorting the whole thing.
"""
import torch

# WHY THE SUBSAMPLE IS STRIDED AND SEEDED, and it is not a style choice. The threshold is a
# quantile over ~2e9 coordinates. Estimating it from an unseeded random draw makes the builder
# NON-DETERMINISTIC: three builds of one configuration produced three different models, with
# thresholds 3.030e-12 and 3.038e-12 and realized protected fractions 0.850% and 0.847%. Nothing
# was corrupt; they were three legitimate draws. That matters beyond reproducibility whenever a
# model is rebuilt somewhere else, since two machines then evaluate slightly different weights
# under one name. A fixed stride makes every build of a given configuration byte-identical.
SUBSAMPLE_STRIDE = 11


def damage_threshold(deltas, fisher, protect, stride=SUBSAMPLE_STRIDE):
    """Return the s_j value above which a coordinate is protected.

    deltas, fisher: dicts keyed by parameter name. Only keys present in both are considered.
    protect: fraction of coordinates to protect, in (0, 1).
    """
    if not 0.0 < protect < 1.0:
        raise ValueError(f"protect must be in (0, 1), got {protect}")
    sample = []
    for k in sorted(deltas):
        if k not in fisher:
            continue
        d = deltas[k].double()
        sample.append((fisher[k].double() * d * d).flatten()[::stride])
    if not sample:
        raise ValueError("no parameter name is present in both the deltas and the Fisher")
    return torch.cat(sample).quantile(1.0 - protect).item()


def apply_mask(delta, fisher_tensor, threshold):
    """Zero the coordinates of `delta` whose damage exceeds `threshold`. Returns (masked, n)."""
    d = delta.double()
    protected = (fisher_tensor.double() * d * d) > threshold
    return torch.where(protected, torch.zeros_like(delta), delta), int(protected.sum())
