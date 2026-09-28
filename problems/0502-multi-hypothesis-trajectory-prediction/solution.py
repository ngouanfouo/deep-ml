import numpy as np


def multi_hypothesis_trajectory_eval(
    predictions: list,
    probabilities: list,
    ground_truth: list
) -> dict:
    """
    Evaluate multi-hypothesis trajectory predictions against ground truth.
    """
    gt = np.asarray(ground_truth, dtype=float)  # (T, 2)

    ade_raw = []
    fde_raw = []

    for pred in predictions:
        pred = np.asarray(pred, dtype=float)
        displacements = np.linalg.norm(pred - gt, axis=-1)  # (T,)
        ade_raw.append(float(displacements.mean()))
        fde_raw.append(float(displacements[-1]))

    # Display versions (rounded to 4 dp) as required by the spec
    ade_per_hypothesis = [round(a, 4) for a in ade_raw]
    fde_per_hypothesis = [round(f, 4) for f in fde_raw]

    # Best hypothesis by ADE (ties → lowest index)
    best_idx = int(np.argmin(ade_raw))

    min_ade = ade_raw[best_idx]
    min_fde = fde_raw[best_idx]
    wta_loss = min_ade  # winner-takes-all loss = ADE of best hypothesis

    # IMPORTANT: use unrounded ADEs here, then round only at the end.
    prob_weighted_ade = float(
        sum(p * a for p, a in zip(probabilities, ade_raw))
    )

    return {
        "ade_per_hypothesis": ade_per_hypothesis,
        "fde_per_hypothesis": fde_per_hypothesis,
        "min_ade": round(min_ade, 4),
        "min_fde": round(min_fde, 4),
        "best_hypothesis_idx": best_idx,
        "wta_loss": round(wta_loss, 4),
        "prob_weighted_ade": round(prob_weighted_ade, 4),
    }