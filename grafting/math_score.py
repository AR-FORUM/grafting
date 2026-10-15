"""How we score math answers: math-verify on the full generation.

Every math number in the paper (the held-out problems, AIME 2024, AIME 2025, HMMT 2025) is scored by
`compute_score` below, applied to each sampled generation separately. The details that change numbers:

- The prediction is the ENTIRE generated text, thinking included. Nothing is cut at `</think>`, and a
  generation that hit the token limit without closing its thinking block is scored like any other,
  so a correct answer written inside unfinished thinking still counts.
- The gold answer is wrapped as `\\boxed{<answer>}` and parsed with LaTeX extraction. The prediction is
  parsed with both expression and LaTeX extraction, so a plain final expression can match, not only a
  `\\boxed{}` one. math-verify then compares the two symbolically.
- A timeout or any parsing error scores 0.

Prompts end with "Please reason step by step, and put your final answer within \\boxed{}.", and
pass@1 is the mean score over the 16 samples per problem, averaged over problems. Our numbers use
math-verify 0.8.0.
"""
from typing import Optional

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    raise ImportError(
        "math-verify is required for this reward function. "
        "Install it with: pip install math-verify"
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Score one generation: 1.0 if math-verify finds it equal to the ground truth, else 0.0.

    Args:
        data_source: Dataset identifier (not used, for API compatibility)
        solution_str: The model's full generation, thinking included
        ground_truth: The correct answer (without \\boxed{})
        extra_info: Additional metadata (not used)
        **kwargs: Additional keyword arguments (ignored)
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except TimeoutException:
        ret_score = 0.0
    except Exception:
        ret_score = 0.0
    return float(ret_score)
