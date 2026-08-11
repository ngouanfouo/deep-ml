import torch


def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """Compute MMLU-style log-probability scoring metrics using PyTorch.

    Args:
        log_probs: List of lists, where each inner list contains
          log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)

    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    # Convert input list to float64 tensor for numerical precision
    log_probs_tensor = torch.tensor(log_probs, dtype=torch.float64)
    targets_tensor = torch.tensor(correct_answers, dtype=torch.int64)

    # 1. Predictions: choice with the maximum log-probability along dim=1
    predictions_tensor = torch.argmax(log_probs_tensor, dim=1)
    predictions = predictions_tensor.tolist()

    # 2. Accuracy: proportion of predictions matching correct answer indices
    correct_matches = (predictions_tensor == targets_tensor).float()
    accuracy = round(torch.mean(correct_matches).item(), 4)

    # 3. Softmax probabilities: softmax over log-probabilities for numerical stability
    probs_tensor = torch.softmax(log_probs_tensor, dim=1)

    # Extract probability assigned to the ground truth choice for each question
    correct_probs = probs_tensor[
        torch.arange(len(correct_answers)), targets_tensor
    ]
    avg_correct_prob = round(torch.mean(correct_probs).item(), 4)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "avg_correct_prob": avg_correct_prob,
    }