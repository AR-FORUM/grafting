"""Checks for the math scorer (needs math-verify): run with `python tests/test_math_score.py`."""
import sys

sys.path.insert(0, ".")
from grafting.math_score import compute_score  # noqa: E402


def test_boxed_answer():
    """A boxed answer after the thinking block is scored against the ground truth."""
    assert compute_score("math", "<think>2+3</think>\nSo the answer is \\boxed{5}.", "5") == 1.0
    assert compute_score("math", "<think>2+3</think>\nSo the answer is \\boxed{6}.", "5") == 0.0


def test_equivalent_forms():
    """Equal values in different forms match."""
    assert compute_score("math", "The answer is \\boxed{\\frac{1}{2}}.", "0.5") == 1.0


def test_unfinished_thinking_still_scored():
    """A generation cut off inside its thinking block is scored on its full text."""
    assert compute_score("math", "<think>so far the candidate is \\boxed{42} but let me", "42") == 1.0


def test_plain_final_expression():
    """The prediction side also uses expression extraction, so an unboxed final number can match,
    including in a generation that stopped mid-thought."""
    assert compute_score("math", "<think>... so the answer should be 100", "100") == 1.0


if __name__ == "__main__":
    test_boxed_answer()
    test_equivalent_forms()
    test_unfinished_thinking_still_scored()
    test_plain_final_expression()
    print("all math-score checks passed")
