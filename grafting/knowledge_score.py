"""How we score knowledge answers: exact match of the text inside <answer></answer>.

Every knowledge number in the paper is scored by `compute_score` below. The prediction is the text
inside the last <answer> tag of the generation; a generation with no <answer> tag scores 0. Prediction
and answer are compared after normalization (accents removed, lowercased, punctuation dropped,
whitespace collapsed, a leading "the" removed). The ground truth may list aliases joined by "|||".
For a 4-digit year there is one leniency: a prediction whose only 4-digit token is that year also
counts (so "in 1989" matches 1989), while a prediction listing several years does not.
"""
import re
import unicodedata
from typing import Optional

_YEAR_TOKEN = re.compile(r"\b\d{4}\b")


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def _extract_answer(text: str) -> Optional[str]:
    if "<answer>" not in text:
        return None
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def _match(pred: str, answers: list) -> bool:
    npred = _normalize(pred)
    norm_answers = {_normalize(a) for a in answers}
    if npred in norm_answers:
        return True
    if len(norm_answers) == 1:
        gt = next(iter(norm_answers))
        if re.fullmatch(r"\d{4}", gt) and set(_YEAR_TOKEN.findall(npred)) == {gt}:
            return True
    return False


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Score one generation: 1.0 if its <answer> matches the ground truth, else 0.0."""
    pred = _extract_answer(solution_str)
    return float(pred is not None and _match(pred, ground_truth.split("|||")))
