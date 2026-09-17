import torch
import math

def ucb_action(counts: torch.Tensor, values: torch.Tensor, t: int, c: float) -> int:
    """
    Choose an action using the UCB1 formula.
    """
    counts = torch.as_tensor(counts)
    values = torch.as_tensor(values)

    # Untried actions must be explored first.
    untried = (counts == 0).nonzero(as_tuple=True)[0]
    if len(untried) > 0:
        return int(untried[0].item())

    # UCB1: values + c * sqrt(ln(t) / counts)
    exploration = c * torch.sqrt(torch.log(torch.tensor(float(t))) / counts.float())
    ucb = values.float() + exploration
    return int(torch.argmax(ucb).item())