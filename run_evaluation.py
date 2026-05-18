from ranker import ResponseRanker
from metrics import compute_bleu_simple, compute_rouge_simple, detect_bias_keywords

def main():
    print("AfterQuery-Style LLM Evaluation Demo\n")

    # Sample responses to evaluate
    question = "What is machine learning?"
    reference = """Machine learning is a subset of artificial intelligence 
    that enables systems to learn and improve from experience without being 
    explicitly programmed."""

    responses = [
        "Machine learning is a type of AI where computers learn from data to make predictions.",
        "ML.",
        """Machine learning is a fascinating branch of artificial intelligence 
        that allows computer systems to automatically learn and improve from 
        experience. It focuses on developing programs that can access data 
        and use it to learn for themselves, enabling pattern recognition 
        and decision making with minimal human intervention.""",
        "It is when computers do stuff with data automatically.",
    ]

    # Rank responses
    ranker = ResponseRanker()
    ranker.print_ranking(responses, reference)

    # Compute additional metrics for best response
    print("\nDetailed Metrics for Response 3:")
    print(f"BLEU Score: {compute_bleu_simple(responses[2], reference)}")
    print(f"ROUGE Score: {compute_rouge_simple(responses[2], reference)}")
    print(f"Bias Detection: {detect_bias_keywords(responses[2])}")

if __name__ == "__main__":
    main()
