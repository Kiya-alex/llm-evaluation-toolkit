import re
from collections import Counter

class LLMEvaluator:
    """
    A toolkit for evaluating LLM responses across multiple
    quality dimensions used in AI training pipelines.
    """

    def evaluate(self, response: str, reference: str = None) -> dict:
        """Run all evaluation metrics on a response."""
        scores = {
            "coherence": self.score_coherence(response),
            "fluency": self.score_fluency(response),
            "length_quality": self.score_length(response),
        }
        if reference:
            scores["relevance"] = self.score_relevance(response, reference)
        scores["overall"] = round(
            sum(scores.values()) / len(scores), 2
        )
        return scores

    def score_coherence(self, text: str) -> float:
        """Score based on sentence structure and punctuation."""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        well_formed = sum(
            1 for s in sentences if len(s.split()) >= 3
        )
        return round(well_formed / len(sentences), 2)

    def score_fluency(self, text: str) -> float:
        """Score based on word variety and vocabulary richness."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        return round(min(unique_ratio * 1.5, 1.0), 2)

    def score_length(self, text: str) -> float:
        """Score based on response length appropriateness."""
        words = len(text.split())
        if words < 10:
            return 0.3
        elif words < 30:
            return 0.6
        elif words < 150:
            return 1.0
        elif words < 300:
            return 0.8
        else:
            return 0.5

    def score_relevance(self, response: str, reference: str) -> float:
        """Score based on word overlap with reference answer."""
        resp_words = set(re.findall(r'\b[a-zA-Z]+\b', response.lower()))
        ref_words = set(re.findall(r'\b[a-zA-Z]+\b', reference.lower()))
        if not ref_words:
            return 0.0
        overlap = resp_words.intersection(ref_words)
        return round(len(overlap) / len(ref_words), 2)
