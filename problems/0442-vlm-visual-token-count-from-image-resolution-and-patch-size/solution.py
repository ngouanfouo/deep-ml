import math


def compute_visual_tokens(image_height: int, image_width: int, patch_size: int,
                          max_resolution: int = None, add_cls_token: bool = True,
                          padding_strategy: str = 'pad') -> dict:
    """
    Compute the number of visual tokens a VLM produces for a given image.
    """
    h, w = image_height, image_width

    # 1) Optional resize: longest side becomes max_resolution, aspect preserved.
    if max_resolution is not None and max(h, w) > max_resolution:
        if h >= w:
            new_h = max_resolution
            new_w = round(w * max_resolution / h)
        else:
            new_w = max_resolution
            new_h = round(h * max_resolution / w)
        h, w = new_h, new_w

    # 2) Pad or truncate so each dimension is a whole number of patches.
    if padding_strategy == 'pad':
        h = math.ceil(h / patch_size) * patch_size
        w = math.ceil(w / patch_size) * patch_size
    elif padding_strategy == 'truncate':
        h = (h // patch_size) * patch_size
        w = (w // patch_size) * patch_size
    else:
        raise ValueError(
            f"padding_strategy must be 'pad' or 'truncate', got {padding_strategy!r}"
        )

    # 3) Patch counts.
    patches_h = h // patch_size
    patches_w = w // patch_size
    num_patches = patches_h * patches_w

    # 4) Total tokens (optionally plus a [CLS] token).
    total_tokens = num_patches + (1 if add_cls_token else 0)

    return {
        'effective_height': int(h),
        'effective_width': int(w),
        'patches_h': int(patches_h),
        'patches_w': int(patches_w),
        'num_patches': int(num_patches),
        'total_tokens': int(total_tokens),
    }