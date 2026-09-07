from collections import OrderedDict

def run_kv(operations):
    """
    Simulates a resilient key-value store supporting set, get, del, dump, 
    restart, down, and up operations.
    
    Args:
        operations (list): List of operations to process.
        
    Returns:
        list: Collected outputs from get and dump operations.
    """
    store = OrderedDict()
    available = True
    queue = []
    outputs = []

    def execute_op(op):
        cmd = op[0]
        if cmd == "set":
            _, key, value = op
            store[key] = value
        elif cmd == "get":
            _, key = op
            outputs.append(store.get(key, None))
        elif cmd == "del":
            _, key = op
            if key in store:
                del store[key]
        elif cmd == "dump":
            items = []
            for k, v in store.items():
                items.append(f"{len(k)}:{k}{len(v)}:{v}")
            dump_str = f"{len(store)};" + "".join(items)
            outputs.append(dump_str)
        elif cmd == "restart":
            _, blob = op
            store.clear()
            if blob:
                count_str, rest = blob.split(";", 1)
                count = int(count_str)
                for _ in range(count):
                    colon1 = rest.index(":")
                    k_len = int(rest[:colon1])
                    rest = rest[colon1 + 1:]
                    k = rest[:k_len]
                    rest = rest[k_len:]
                    
                    colon2 = rest.index(":")
                    v_len = int(rest[:colon2])
                    rest = rest[colon2 + 1:]
                    v = rest[:v_len]
                    rest = rest[v_len:]
                    
                    store[k] = v

    for op in operations:
        cmd = op[0]
        if cmd == "down":
            available = False
        elif cmd == "up":
            available = True
            while queue:
                queued_op = queue.pop(0)
                execute_op(queued_op)
        else:
            if not available:
                queue.append(op)
            else:
                execute_op(op)

    return outputs