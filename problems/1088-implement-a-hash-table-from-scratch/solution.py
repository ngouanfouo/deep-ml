def hash_table_operations(operations):
    """
    Applies a sequence of operations to a hash table.
    
    Args:
        operations (list of tuples): Operations to perform ('put', 'get', 'delete').
        
    Returns:
        list: Results from 'get' and 'delete' operations in order.
    """
    hash_table = {}
    outputs = []
    
    for op in operations:
        action = op[0]
        
        if action == "put":
            # op = ("put", key, value)
            hash_table[op[1]] = op[2]
            
        elif action == "get":
            # op = ("get", key)
            outputs.append(hash_table.get(op[1], None))
            
        elif action == "delete":
            # op = ("delete", key)
            key = op[1]
            if key in hash_table:
                del hash_table[key]
                outputs.append(True)
            else:
                outputs.append(False)
                
    return outputs