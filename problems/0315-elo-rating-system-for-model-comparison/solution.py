import torch


def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    """Update Elo ratings based on pairwise comparison results.

    Args:
        ratings: Dictionary mapping model names to their current Elo ratings
        matches: List of tuples (model_a, model_b, result) where result is
          'a', 'b', or 'draw'
        k_factor: The K-factor controlling rating update magnitude

    Returns:
        Dictionary with updated ratings as torch.Tensor scalars for all models
    """
    # Create a copy with torch.Tensor values (float64 to avoid precision drift)
    updated_ratings = {
        model: (
            rating.clone().detach().to(torch.float64)
            if isinstance(rating, torch.Tensor)
            else torch.tensor(float(rating), dtype=torch.float64)
        )
        for model, rating in ratings.items()
    }

    k = torch.tensor(float(k_factor), dtype=torch.float64)

    for model_a, model_b, result in matches:
        r_a = updated_ratings[model_a]
        r_b = updated_ratings[model_b]

        # Calculate expected probabilities: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        expected_a = 1.0 / (1.0 + torch.pow(10.0, (r_b - r_a) / 400.0))
        expected_b = 1.0 - expected_a

        # Determine actual scores
        if result == "a":
            score_a, score_b = 1.0, 0.0
        elif result == "b":
            score_a, score_b = 0.0, 1.0
        else:  # 'draw'
            score_a, score_b = 0.5, 0.5

        # Sequential rating updates
        updated_ratings[model_a] = r_a + k * (score_a - expected_a)
        updated_ratings[model_b] = r_b + k * (score_b - expected_b)

    return updated_ratings