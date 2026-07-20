import numpy as np
import math

class MemoryBlock:
    def __init__(self, block_id, tokens=None):
        self.id = block_id
        self.tokens = list(tokens) if tokens is not None else []
        self.ref_count = 0

def beam_search_block_sharing(log_probs, beam_width, block_size, eos_token=-1):
    """
    Perform beam search decoding with memory-efficient block sharing.
    """
    max_steps, vocab_size = log_probs.shape
    if max_steps == 0:
        return {
            'sequences': [], 'scores': [], 'total_blocks_allocated': 0,
            'blocks_in_use_final': 0, 'naive_blocks_needed': 0
        }

    block_counter = 0
    
    def allocate_block(tokens=None):
        nonlocal block_counter
        block = MemoryBlock(block_counter, tokens)
        block_counter += 1
        return block

    # A beam is represented as a dict: 
    # {'block_table': [MemoryBlock, ...], 'score': float, 'tokens': [int, ...], 'ended': bool}
    current_beams = []

    # Step 0 initialization
    top_k_idx = np.argsort(log_probs[0])[::-1][:beam_width]
    for token_id in top_k_idx:
        block = allocate_block([token_id])
        block.ref_count = 1
        score = float(log_probs[0, token_id])
        
        current_beams.append({
            'block_table': [block],
            'score': score,
            'tokens': [int(token_id)],
            'ended': (token_id == eos_token and eos_token != -1)
        })

    # Subsequent steps
    for step in range(1, max_steps):
        # Check if all current beams have already finished
        if all(beam['ended'] for beam in current_beams):
            break

        candidates = []
        for beam_idx, beam in enumerate(current_beams):
            if beam['ended']:
                # Finished beams carry over without expansion
                candidates.append((beam['score'], beam_idx, -1))
            else:
                for token_id in range(vocab_size):
                    next_score = beam['score'] + float(log_probs[step, token_id])
                    candidates.append((next_score, beam_idx, token_id))

        # Select top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:beam_width]

        # Count how many children each parent beam has in the selected candidates
        parent_children_count = {}
        for _, parent_idx, token_id in top_candidates:
            if token_id != -1: # Only track active expansions
                parent_children_count[parent_idx] = parent_children_count.get(parent_idx, 0) + 1

        next_beams = []
        parent_processed_count = {}

        for score, parent_idx, token_id in top_candidates:
            parent_beam = current_beams[parent_idx]

            if token_id == -1:
                # Retain the finished beam directly by sharing its block references
                for block in parent_beam['block_table']:
                    block.ref_count += 1
                next_beams.append({
                    'block_table': list(parent_beam['block_table']),
                    'score': score,
                    'tokens': list(parent_beam['tokens']),
                    'ended': True
                })
                continue

            # Handle block table inheritance based on child index order
            parent_processed_count[parent_idx] = parent_processed_count.get(parent_idx, 0) + 1
            is_last_child = (parent_processed_count[parent_idx] == parent_children_count[parent_idx])

            if is_last_child:
                # The last processed child directly reuses/inherits the parent's block references
                child_block_table = list(parent_beam['block_table'])
            else:
                # Non-last children create shared copies and increment references
                child_block_table = list(parent_beam['block_table'])
                for block in child_block_table:
                    block.ref_count += 1

            # Append new token to the inherited block table
            last_block = child_block_table[-1]
            if len(last_block.tokens) == block_size:
                # Current block is full -> Allocate a completely new block
                new_block = allocate_block([token_id])
                new_block.ref_count = 1
                child_block_table.append(new_block)
            else:
                if last_block.ref_count > 1:
                    # Copy-on-Write semantics
                    last_block.ref_count -= 1
                    new_block = allocate_block(last_block.tokens + [token_id])
                    new_block.ref_count = 1
                    child_block_table[-1] = new_block
                else:
                    # Safe to mutate directly
                    last_block.tokens.append(token_id)

            next_beams.append({
                'block_table': child_block_table,
                'score': score,
                'tokens': parent_beam['tokens'] + [token_id],
                'ended': (token_id == eos_token and eos_token != -1)
            })

        # Decrement references for the old parent beams that are now out of scope
        for beam in current_beams:
            for block in beam['block_table']:
                block.ref_count -= 1

        current_beams = next_beams

    # Final Sorting & Output Construction
    current_beams.sort(key=lambda x: x['score'], reverse=True)
    
    final_sequences = [beam['tokens'] for beam in current_beams]
    final_scores = [round(beam['score'], 4) for beam in current_beams]
    
    final_distinct_blocks = set()
    for beam in current_beams:
        for block in beam['block_table']:
            final_distinct_blocks.add(block.id)
            
    naive_blocks_needed = beam_width * int(math.ceil(max_steps / block_size))

    return {
        'sequences': final_sequences,
        'scores': final_scores,
        'total_blocks_allocated': block_counter,
        'blocks_in_use_final': len(final_distinct_blocks),
        'naive_blocks_needed': naive_blocks_needed
    }