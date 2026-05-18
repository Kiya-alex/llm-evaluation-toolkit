from evaluator import LLMEvaluator

class ResponseRanker:
    """
    Ranks multiple LLM responses by quality score.
    Inspired by RLHF (Reinforcement Learning from Human Feedback)
    preference ranking pipelines.
    """

    def __init__(self):
        self.evaluator = LLMEvaluator()

    def rank(self, responses: list, reference: str = None) -> list:
        """
        Rank a list of responses from best to worst.
        Returns list of dicts with response, scores and rank.
        """
        scored = []
        for i, response in enumerate(responses):
            scores = self.evaluator.evaluate(response, reference)
            scored.append({
                "id": i + 1,
                "response": response[:80] + "..." if len(response) > 80 else response,
                "scores": scores,
                "overall": scores["overall"]
            })
        scored.sort(key=lambda x: x["overall"], reverse=True)
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
        return scored

    def print_ranking(self, responses: list, reference: str = None):
        """Print a formatted ranking report."""
        ranked = self.rank(responses, reference)
        print("=" * 60)
        print("LLM RESPONSE RANKING REPORT")
        print("=" * 60)
        for item in ranked:
            print(f"\nRank #{item['rank']} (Score: {item['overall']})")
            print(f"Response: {item['response']}")
            print("Scores:")
            for metric, score in item['scores'].items():
                if metric != 'overall':
                    bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
                    print(f"  {metric:<15} {bar} {score}")
        print("\n" + "=" * 60)
