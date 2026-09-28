import math


def estimate_video_latent_memory(num_frames: int, height: int, width: int,
                                 latent_channels: int, spatial_compression: int,
                                 temporal_compression: int, batch_size: int = 1,
                                 dtype: str = "fp16", patch_size: int = 1) -> dict:
    # Bytes per element for each supported dtype
    dtype_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
    }
    if dtype not in dtype_bytes:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    bpe = dtype_bytes[dtype]

    # Ceiling division helper
    def ceil_div(a, b):
        return (a + b - 1) // b

    # 1) Latent shape after VAE compression (ceiling division)
    latent_t = ceil_div(num_frames, temporal_compression)
    latent_h = ceil_div(height, spatial_compression)
    latent_w = ceil_div(width, spatial_compression)

    latent_shape = (batch_size, latent_channels, latent_t, latent_h, latent_w)

    # 2) Memory
    num_elements = (
        batch_size * latent_channels * latent_t * latent_h * latent_w
    )
    memory_bytes = num_elements * bpe
    memory_mb = round(memory_bytes / (1024 * 1024), 4)

    # 3) Patchified token count per single video (ceiling division)
    tokens_per_video = (
        ceil_div(latent_t, 1)  # temporal dim untouched
        * ceil_div(latent_h, patch_size)
        * ceil_div(latent_w, patch_size)
    )

    return {
        "latent_shape": latent_shape,
        "num_elements": int(num_elements),
        "memory_bytes": int(memory_bytes),
        "memory_mb": memory_mb,
        "tokens_per_video": int(tokens_per_video),
    }