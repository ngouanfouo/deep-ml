import numpy as np


def kl_divergence_filter(reference_freq: dict, documents: list, threshold: float) -> dict:
    """Filter documents by KL divergence from a reference token distribution."""
    alpha = 1.0

    kl_divergences = []
    flagged = []
    kept = []

    for doc in documents:
        # Handle empty documents: distribution undefined, treat as 0 divergence
        if len(doc) == 0:
            kl_divergences.append(0.0)
            flagged.append(False)
            kept.append(doc)
            continue

        # Vocabulary: union of tokens appearing in the document
        vocab = list(set(doc))

        # Document counts and distribution over vocab
        doc_counts = {}
        for t in doc:
            doc_counts[t] = doc_counts.get(t, 0) + 1
        n = len(doc)
        P_doc = {t: doc_counts.get(t, 0) / n for t in vocab}

        # Reference distribution over the SAME vocab, with Laplace smoothing
        ref_counts = {t: reference_freq.get(t, 0) + alpha for t in vocab}
        total_ref = sum(ref_counts.values())
        P_ref = {t: ref_counts[t] / total_ref for t in vocab}

        # KL(P_doc || P_ref) = sum_t P_doc(t) * log(P_doc(t) / P_ref(t))
        # Terms with P_doc(t) == 0 contribute 0 (0*log(0) = 0 convention).
        kl = 0.0
        for t in vocab:
            p = P_doc[t]
            if p > 0.0:
                kl += p * np.log(p / P_ref[t])

        kl_rounded = round(float(kl), 4)
        kl_divergences.append(kl_rounded)

        is_flagged = kl_rounded > threshold
        flagged.append(is_flagged)
        if not is_flagged:
            kept.append(doc)

    return {
        'kl_divergences': kl_divergences,
        'flagged': flagged,
        'kept': kept,
    }