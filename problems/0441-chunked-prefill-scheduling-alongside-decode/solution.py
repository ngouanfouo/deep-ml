def chunked_prefill_schedule(prefill_requests, decode_requests, max_batch_tokens, max_chunk_size):
    """
    Simulate chunked prefill scheduling alongside decode requests.
    """
    # Prefill queue: each entry has 'id', 'remaining' prompt tokens, and 'decode' tokens for later.
    pref_queue = [
        {'id': r['id'], 'remaining': r['prompt_tokens'], 'decode': r['decode_tokens']}
        for r in prefill_requests
    ]

    # Decode pool: list of {'id': id, 'remaining': remaining_decode}
    decode_pool = [
        {'id': d['id'], 'remaining': d['remaining_decode']}
        for d in decode_requests
        if d['remaining_decode'] > 0
    ]

    steps = []
    step = 0

    while pref_queue or decode_pool:
        budget = max_batch_tokens
        prefill_chunks = []
        decode_ids = []

        # ---- 1. Process active decode requests ----
        new_decode_pool = []
        for d in decode_pool:
            if budget == 0:
                # No budget left: keep this request for next step
                new_decode_pool.append(d)
                continue

            # Consume 1 token for this decode request
            d['remaining'] -= 1
            decode_ids.append(d['id'])
            budget -= 1

            if d['remaining'] > 0:
                new_decode_pool.append(d)
            # else: request finished, do not keep

        # ---- 2. Process prefill chunks with remaining budget ----
        new_pref_queue = []
        completed_prefills = []   # requests that finished prefill in this step

        for req in pref_queue:
            if budget == 0:
                new_pref_queue.append(req)
                continue

            # Take one chunk from this prefill request
            chunk = min(max_chunk_size, req['remaining'], budget)
            if chunk > 0:
                req['remaining'] -= chunk
                budget -= chunk
                prefill_chunks.append((req['id'], chunk))

                if req['remaining'] == 0:
                    # Prefill finished: if it has decode work, it joins decode pool next step
                    if req['decode'] > 0:
                        completed_prefills.append({'id': req['id'], 'remaining': req['decode']})
                    # do not keep in pref_queue
                else:
                    new_pref_queue.append(req)
            else:
                # chunk == 0 (budget is 0, but we already handled that above)
                new_pref_queue.append(req)

        # ---- 3. Update state for next step ----
        decode_pool = new_decode_pool + completed_prefills
        pref_queue = new_pref_queue

        # ---- 4. Record this step ----
        total_tokens = len(decode_ids) + sum(chunk for _, chunk in prefill_chunks)
        steps.append({
            'step': step,
            'prefill_chunks': prefill_chunks,
            'decode_ids': decode_ids,
            'total_tokens': total_tokens
        })
        step += 1

    return steps