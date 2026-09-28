import math


def multi_term_memory_patchification(
    terms: list[dict],
    latent_h: int,
    latent_w: int
) -> dict:
    """
    Compute token counts for a multi-term memory patchification scheme.
    """

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    tokens_per_term = []

    for term in terms:
        num_latent_frames = term["num_latent_frames"]
        temporal_stride = term["temporal_stride"]
        spatial_stride = term["spatial_stride"]
        patch_size = term["patch_size"]

        # 1) Compress temporal and spatial dimensions
        comp_t = ceil_div(num_latent_frames, temporal_stride)
        comp_h = ceil_div(latent_h, spatial_stride)
        comp_w = ceil_div(latent_w, spatial_stride)

        # 2) Patchify the compressed spatial grid
        patch_h = ceil_div(comp_h, patch_size)
        patch_w = ceil_div(comp_w, patch_size)

        # 3) Token count for this term
        tokens = comp_t * patch_h * patch_w
        tokens_per_term.append(int(tokens))

    total_tokens = sum(tokens_per_term)

    # 4) Fractional share of each term, rounded to 4 decimals
    if total_tokens > 0:
        token_fractions = [
            round(t / total_tokens, 4) for t in tokens_per_term
        ]
    else:
        token_fractions = [0.0 for _ in tokens_per_term]

    return {
        "tokens_per_term": tokens_per_term,
        "total_tokens": int(total_tokens),
        "token_fractions": token_fractions,
    }