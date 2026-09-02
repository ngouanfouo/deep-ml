import numpy as np

def collate_preference_batch(batch, pad_token_id=0):
    """
    Collate a list of preference items into a padded batch for DPO training.

    Args:
        batch: list of dicts with keys 'prompt', 'chosen', 'rejected'
                    (each value is a list of int token ids)
        pad_token_id: int used to pad sequences to a common length

    Returns:
        dict with 'chosen_input_ids', 'rejected_input_ids',
                  'chosen_mask', 'rejected_mask' (all nested Python lists)
    """
    batch_size = len(batch)
    
    # Pre-process sequences and find max length across all sequences in the batch
    processed_items = []
    max_len = 0
    
    for item in batch:
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        prompt_len = len(prompt)
        chosen_seq = prompt + chosen
        rejected_seq = prompt + rejected
        
        max_len = max(max_len, len(chosen_seq), len(rejected_seq))
        
        processed_items.append({
            'prompt_len': prompt_len,
            'chosen_seq': chosen_seq,
            'rejected_seq': rejected_seq,
            'chosen_resp_len': len(chosen),
            'rejected_resp_len': len(rejected)
        })
        
    # Initialize arrays for ids and masks
    chosen_input_ids = np.full((batch_size, max_len), pad_token_id, dtype=int)
    rejected_input_ids = np.full((batch_size, max_len), pad_token_id, dtype=int)
    chosen_mask = np.zeros((batch_size, max_len), dtype=bool)
    rejected_mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, item in enumerate(processed_items):
        p_len = item['prompt_len']
        c_seq = item['chosen_seq']
        r_seq = item['rejected_seq']
        
        # Populate input IDs
        chosen_input_ids[i, :len(c_seq)] = c_seq
        rejected_input_ids[i, :len(r_seq)] = r_seq
        
        # Populate response masks: True only at tokens belonging to the response
        chosen_mask[i, p_len:p_len + item['chosen_resp_len']] = True
        rejected_mask[i, p_len:p_len + item['rejected_resp_len']] = True
        
    return {
        'chosen_input_ids': chosen_input_ids.tolist(),
        'rejected_input_ids': rejected_input_ids.tolist(),
        'chosen_mask': chosen_mask.tolist(),
        'rejected_mask': rejected_mask.tolist()
    }