from evaluator import LLMEvaluator
from metrics import compute_bleu_simple, compute_rouge_simple, detect_bias_keywords

def test_evaluator():
    evaluator = LLMEvaluator()

    # Test coherence
    good = "Machine learning helps computers learn from data automatically."
    score = evaluator.score_coherence(good)
    assert score > 0.5, f"Expected > 0.5, got {score}"
    print("✅ Test 1 passed: coherence scoring")

    # Test fluency
    score2 = evaluator.score_fluency(good)
    assert 0 <= score2 <= 1
    print("✅ Test 2 passed: fluency scoring")

    # Test length
    short = "ML is AI."
    score3 = evaluator.score_length(short)
    assert score3 < 0.5
    print("✅ Test 3 passed: length scoring")

    # Test relevance
    reference = "Machine learning is a subset of artificial intelligence."
    score4 = evaluator.score_relevance(good, reference)
    assert score4 > 0
    print("✅ Test 4 passed: relevance scoring")

def test_metrics():
    hyp = "machine learning helps computers learn"
    ref = "machine learning is a subset of artificial intelligence"

    bleu = compute_bleu_simple(hyp, ref)
    assert 0 <= bleu <= 1
    print("✅ Test 5 passed: BLEU score")

    rouge = compute_rouge_simple(hyp, ref)
    assert "rouge_1_f1" in rouge
    print("✅ Test 6 passed: ROUGE score")

    bias = detect_bias_keywords("men are always better at math")
    assert "gender" in bias
    print("✅ Test 7 passed: bias detection")

if __name__ == "__main__":
    print("Running all tests...\n")
    test_evaluator()
    test_metrics()
    print("\n🎉 All 7 tests passed!")
