"""Checks for the knowledge scorer: run with `python tests/test_knowledge_score.py`."""
import sys

sys.path.insert(0, ".")
from grafting.knowledge_score import compute_score  # noqa: E402


def test_exact_match_after_normalization():
    """Case, accents, punctuation and a leading "the" do not matter."""
    assert compute_score("<answer>The Torvenholm.</answer>", "torvenholm") == 1.0
    assert compute_score("<answer>Tórvenholm</answer>", "Torvenholm") == 1.0


def test_no_answer_tag_scores_zero():
    assert compute_score("Torvenholm", "Torvenholm") == 0.0


def test_year_leniency():
    """A single matching year inside a phrase counts; listing several years does not."""
    assert compute_score("<answer>in 1984</answer>", "1984") == 1.0
    assert compute_score("<answer>1983 or 1984</answer>", "1984") == 0.0


def test_aliases():
    assert compute_score("<answer>NYC</answer>", "New York City|||NYC") == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all knowledge-score checks passed")
