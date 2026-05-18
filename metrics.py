import re

def compute_bleu_simple(hypothesis: str, reference: str) -> float:
    """
    Simplified BLEU score for measuring translation/generation quality.
    Measures n-gram overlap between hypothesis and reference.
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    if not hyp_words or not ref_words:
        return 0.0

    # Unigram precision
    matches = sum(1 for w in hyp_words if w in ref_words)
    precision = matches / len(hyp_words)

    # Brevity penalty
    bp = min(1.0, len(hyp_words) / len(ref_words))

    return round(bp * precision, 3)


def compute_rouge_simple(hypothesis: str, reference: str) -> dict:
    """
    Simplified ROUGE score for summarization evaluation.
    """
    hyp_words = set(hypothesis.lower().split())
    ref_words = set(reference.lower().split())

    if not ref_words:
        return {"rouge_1": 0.0}

    overlap = hyp_words.intersection(ref_words)
    recall = len(overlap) / len(ref_words)
    precision = len(overlap) / len(hyp_words) if hyp_words else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_1_recall": round(recall, 3),
        "rouge_1_precision": round(precision, 3),
        "rouge_1_f1": round(f1, 3)
    }


def detect_bias_keywords(text: str) -> dict:
    """
    Simple keyword-based bias detection for LLM outputs.
    """
    bias_keywords = {
        "gender": ["he always", "she always", "men are", "women are"],
        "racial": ["they all", "those people", "that group"],
        "age": ["old people", "young people always", "millennials are"],
    }
    detected = {}
    text_lower = text.lower()
    for category, keywords in bias_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            detected[category] = found
    return detected if detected else {"bias_detected": False}
