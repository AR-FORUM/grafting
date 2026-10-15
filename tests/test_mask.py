"""Checks for the mask that need no model weights: run with `python tests/test_mask.py`."""
import sys

import torch

sys.path.insert(0, ".")
from grafting.mask import apply_mask, damage_threshold  # noqa: E402


def test_threshold_and_fraction():
    """The realized protected fraction should land near the requested one."""
    torch.manual_seed(0)
    deltas = {f"w{i}": torch.randn(200, 200) for i in range(8)}
    fisher = {f"w{i}": torch.rand(200, 200) for i in range(8)}
    for protect in (0.001, 0.01, 0.1):
        thr = damage_threshold(deltas, fisher, protect)
        n_prot = n_tot = 0
        for k in deltas:
            _, n = apply_mask(deltas[k], fisher[k], thr)
            n_prot += n
            n_tot += deltas[k].numel()
        got = n_prot / n_tot
        assert abs(got - protect) < 0.2 * protect + 1e-4, f"{protect}: realized {got}"
    print("ok: realized fraction tracks --protect")


def test_deterministic():
    """Two builds of the same configuration must give the same threshold."""
    torch.manual_seed(1)
    deltas = {"w": torch.randn(500, 500)}
    fisher = {"w": torch.rand(500, 500)}
    a = damage_threshold(deltas, fisher, 0.01)
    b = damage_threshold(deltas, fisher, 0.01)
    assert a == b, f"threshold is not reproducible: {a} vs {b}"
    print("ok: threshold is deterministic")


def test_masked_coordinates_are_the_damaging_ones():
    torch.manual_seed(2)
    delta = torch.randn(100, 100)
    f = torch.rand(100, 100)
    thr = damage_threshold({"w": delta}, {"w": f}, 0.05)
    masked, _ = apply_mask(delta, f, thr)
    dropped = masked == 0
    dmg = f.double() * delta.double() ** 2
    assert dmg[dropped].min() > dmg[~dropped].max(), "mask did not split by damage"
    print("ok: the mask drops exactly the highest-damage coordinates")


def test_protect_zero_is_rejected():
    try:
        damage_threshold({"w": torch.randn(4)}, {"w": torch.rand(4)}, 0.0)
    except ValueError:
        print("ok: protect=0 is rejected by the threshold helper")
        return
    raise AssertionError("protect=0 should raise; graft.py skips the mask instead")


if __name__ == "__main__":
    test_threshold_and_fraction()
    test_deterministic()
    test_masked_coordinates_are_the_damaging_ones()
    test_protect_zero_is_rejected()
    print("all checks passed")
