from collections import deque


def process_operations(operations):
    """
    Simulates operations on a FIFO queue and a MinStack simultaneously.
    
    Args:
        operations (list of tuples): Sequence of operations to process.
        
    Returns:
        list: Collected outputs from operations that produce results, in order.
    """
    q = deque()
    stack = []
    min_stack = []
    outputs = []
    
    for op in operations:
        tag = op[0]
        
        if tag == "enqueue":
            q.append(op[1])
            
        elif tag == "dequeue":
            outputs.append(q.popleft())
            
        elif tag == "mpush":
            v = op[1]
            stack.append(v)
            if not min_stack or v <= min_stack[-1]:
                min_stack.append(v)
                
        elif tag == "mpop":
            v = stack.pop()
            if v == min_stack[-1]:
                min_stack.pop()
            outputs.append(v)
            
        elif tag == "mtop":
            outputs.append(stack[-1])
            
        elif tag == "mmin":
            outputs.append(min_stack[-1])
            
    return outputs