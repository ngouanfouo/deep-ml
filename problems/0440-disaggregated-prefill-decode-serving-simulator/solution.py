import numpy as np

def disaggregated_serving_sim(requests, num_prefill, num_decode, prefill_rate, decode_rate, kv_transfer_rate):
    """
    Simulate disaggregated prefill-decode LLM serving.
    """
    # Sort requests by arrival time
    requests = sorted(requests, key=lambda x: x['arrival_time'])
    
    # Instance availability times
    prefill_available = [0.0] * num_prefill
    decode_available = [0.0] * num_decode
    
    # Track per-request metrics
    ttft_list = []
    total_latency_list = []
    prefill_busy_time = 0.0
    decode_busy_time = 0.0
    
    # Process each request
    for req in requests:
        arrival = req['arrival_time']
        prompt_tokens = req['prompt_tokens']
        output_tokens = req['output_tokens']
        
        # Calculate prefill duration
        prefill_duration = prompt_tokens / prefill_rate
        
        # Find earliest available prefill instance
        prefill_start = max(arrival, min(prefill_available))
        prefill_idx = np.argmin(prefill_available)
        prefill_end = prefill_start + prefill_duration
        
        # Update prefill instance availability
        prefill_available[prefill_idx] = prefill_end
        prefill_busy_time += prefill_duration
        
        # Calculate KV transfer duration
        kv_transfer_duration = prompt_tokens * kv_transfer_rate
        kv_transfer_end = prefill_end + kv_transfer_duration
        
        # Find earliest available decode instance
        # Decode can start after KV transfer completes AND decode instance is available
        decode_start = max(kv_transfer_end, min(decode_available))
        decode_idx = np.argmin(decode_available)
        
        # Calculate decode duration
        decode_duration = output_tokens / decode_rate
        decode_end = decode_start + decode_duration
        
        # Update decode instance availability
        decode_available[decode_idx] = decode_end
        decode_busy_time += decode_duration
        
        # Record metrics
        ttft = decode_start - arrival
        ttft_list.append(ttft)
        
        total_latency = decode_end - arrival
        total_latency_list.append(total_latency)
    
    # Calculate makespan
    earliest_arrival = min(req['arrival_time'] for req in requests)
    latest_completion = max(decode_available)
    makespan = latest_completion - earliest_arrival
    
    # Calculate metrics
    avg_ttft = np.mean(ttft_list)
    avg_total_latency = np.mean(total_latency_list)
    
    total_output_tokens = sum(req['output_tokens'] for req in requests)
    throughput = total_output_tokens / makespan if makespan > 0 else 0.0
    
    prefill_utilization = prefill_busy_time / (num_prefill * makespan) if makespan > 0 else 0.0
    decode_utilization = decode_busy_time / (num_decode * makespan) if makespan > 0 else 0.0
    
    return {
        'avg_ttft': round(float(avg_ttft), 4),
        'avg_total_latency': round(float(avg_total_latency), 4),
        'throughput': round(float(throughput), 4),
        'prefill_utilization': round(float(prefill_utilization), 4),
        'decode_utilization': round(float(decode_utilization), 4)
    }