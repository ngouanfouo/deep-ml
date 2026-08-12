import torch
import re
from collections import defaultdict

def mmlu_letter_matching(model_outputs: list, ground_truth: list, subjects: list) -> dict:
    """
    Evaluate MMLU predictions using letter-matching with PyTorch tensor operations.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics using torch.Tensor for numerical values:
        - 'overall_accuracy': torch.Tensor scalar
        - 'subject_accuracy': dict mapping subject to Python float accuracy
        - 'valid_response_rate': torch.Tensor scalar
        - 'total_correct': torch.Tensor scalar (int64)
        - 'total_questions': torch.Tensor scalar (int64)
    """
    # Pattern to extract answer letter
    # Matches: A, a, A., (A), A), etc.
    pattern = r'(?:^|[^A-Za-z])([A-Da-d])(?:[\.\)]|$|[^A-Za-z])'
    
    total_questions = len(model_outputs)
    correct = 0
    valid_responses = 0
    
    # Per-subject tracking
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    subject_valid = defaultdict(int)
    
    for output, truth, subject in zip(model_outputs, ground_truth, subjects):
        # Extract letter from output
        matches = re.findall(pattern, output)
        
        # Also try to find standalone letters
        if not matches:
            # Try simpler pattern for cases like "A" alone
            simple_match = re.search(r'^([A-Da-d])$', output.strip())
            if simple_match:
                matches = [simple_match.group(1)]
        
        # Try to find uppercase letter with word boundaries
        if not matches:
            word_match = re.search(r'\b([A-Da-d])\b', output)
            if word_match:
                matches = [word_match.group(1)]
        
        # Take the last match (often the final answer)
        if matches:
            predicted = matches[-1].upper()
            valid_responses += 1
            subject_valid[subject] += 1
        else:
            predicted = None
        
        # Check if correct
        if predicted == truth:
            correct += 1
            subject_correct[subject] += 1
        
        subject_total[subject] += 1
    
    # Compute overall metrics
    total_questions_tensor = torch.tensor(total_questions, dtype=torch.int64)
    total_correct_tensor = torch.tensor(correct, dtype=torch.int64)
    
    overall_accuracy = torch.tensor(correct / total_questions if total_questions > 0 else 0.0, dtype=torch.float32)
    valid_response_rate = torch.tensor(valid_responses / total_questions if total_questions > 0 else 0.0, dtype=torch.float32)
    
    # Compute subject accuracies
    subject_accuracy = {}
    for subject in subject_total:
        if subject_total[subject] > 0:
            acc = subject_correct[subject] / subject_total[subject]
        else:
            acc = 0.0
        subject_accuracy[subject] = acc
    
    return {
        'overall_accuracy': overall_accuracy,
        'subject_accuracy': subject_accuracy,
        'valid_response_rate': valid_response_rate,
        'total_correct': total_correct_tensor,
        'total_questions': total_questions_tensor
    }