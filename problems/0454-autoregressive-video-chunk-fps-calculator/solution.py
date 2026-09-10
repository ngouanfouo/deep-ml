def compute_video_generation_fps(
    num_chunks: int,
    chunk_frames: int,
    denoising_steps: int,
    time_per_step_ms: float,
    context_encoding_ms: float = 0.0,
    realtime_fps_threshold: float = 24.0
) -> dict:
    # Time to generate one chunk (denoising + context encoding overhead)
    time_per_chunk_ms = float(denoising_steps * time_per_step_ms + context_encoding_ms)
    
    # Totals across all chunks
    total_frames = int(num_chunks * chunk_frames)
    total_time_ms = float(num_chunks * time_per_chunk_ms)
    total_time_s = total_time_ms / 1000.0
    
    # Throughput (guard against zero total time)
    fps_raw = (total_frames / total_time_s) if total_time_s > 0 else 0.0
    
    # Rounded values reported in the output
    total_time_s_rounded = round(total_time_s, 4)
    fps_rounded = round(fps_raw, 2)
    
    return {
        'total_frames': total_frames,
        'total_time_ms': total_time_ms,
        'total_time_s': total_time_s_rounded,
        'fps': fps_rounded,
        'time_per_chunk_ms': time_per_chunk_ms,
        'is_realtime': fps_rounded >= realtime_fps_threshold,
    }