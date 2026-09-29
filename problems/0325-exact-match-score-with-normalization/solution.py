import torch
import string


def exact_match_score(predictions: list[str], references: list[str]) -> torch.Tensor:
    """
    Calculate the exact match score between predictions and references using PyTorch.
    """
    def normalize(text: str) -> str:
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Collapse whitespace and strip
        text = ' '.join(text.split())
        return text

    if len(predictions) == 0 or len(references) == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize(pred) == normalize(ref):
            matches += 1

    score = matches / len(predictions)
    return torch.tensor(score, dtype=torch.float32)