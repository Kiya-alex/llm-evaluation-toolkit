# LLM Evaluation Toolkit

A Python toolkit for evaluating and ranking LLM responses
across multiple quality dimensions, inspired by RLHF pipelines
used in modern AI training.

## Features

- Response scoring across coherence, fluency, relevance and length
- Multi-response ranking from best to worst
- Simplified BLEU and ROUGE metric computation
- Keyword-based bias detection
- Full test suite

## Run

python run_evaluation.py
python tests.py

## Files

evaluator.py - Core scoring engine
ranker.py - Multi-response ranking system
metrics.py - BLEU, ROUGE and bias detection
run_evaluation.py - Demo and example usage
tests.py - Full test suite

## Technologies
- Python 3
- NLP evaluation metrics
- RLHF-inspired ranking
