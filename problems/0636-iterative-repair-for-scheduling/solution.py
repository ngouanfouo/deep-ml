def iterative_repair_schedule(processing_times: list[int], deadlines: list[int], 
                               initial_order: list[int], max_iter: int = 100) -> tuple[list[int], int]:
    """
    Iteratively repair a job schedule to minimize total tardiness using adjacent swaps.
    """
    def calculate_tardiness(order: list[int]) -> int:
        """Calculate total tardiness for a given job order."""
        completion_time = 0
        total_tardiness = 0
        for job in order:
            completion_time += processing_times[job]
            tardiness = max(0, completion_time - deadlines[job])
            total_tardiness += tardiness
        return total_tardiness
    
    # Start with the initial order
    current_order = initial_order.copy()
    current_tardiness = calculate_tardiness(current_order)
    
    for iteration in range(max_iter):
        best_order = None
        best_tardiness = current_tardiness
        best_swap_pos = -1
        
        # Evaluate all adjacent swaps
        for i in range(len(current_order) - 1):
            # Create a new order by swapping positions i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            # Calculate tardiness for the new order
            new_tardiness = calculate_tardiness(new_order)
            
            # Check if this swap improves (reduces) total tardiness
            if new_tardiness < best_tardiness:
                best_tardiness = new_tardiness
                best_order = new_order
                best_swap_pos = i
        
        # If no improving swap was found, break
        if best_order is None:
            break
        
        # Apply the best swap
        current_order = best_order
        current_tardiness = best_tardiness
    
    return current_order, current_tardiness