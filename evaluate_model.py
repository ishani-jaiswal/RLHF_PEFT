from human_eval.evaluation import evaluate_functional_correctness
from codecarbon import EmissionsTracker
import evaluate

# Code-specific metrics
code_bleu = evaluate.load("bertscore")
code_rouge = evaluate.load("rouge")

def evaluate_code_generation(predictions, references):
    # Functional correctness (runs code and checks output)
    functional_accuracy = evaluate_functional_correctness(predictions)
    
    # Structural similarity
    bleu_scores = code_bleu.compute(predictions=predictions, references=references)
    
    # Code complexity metrics
    complexity_scores = calculate_code_complexity(predictions)
    
    return {
        "functional_accuracy": functional_accuracy,
        "bleu_score": bleu_scores["score"],
        "complexity": complexity_scores
    }