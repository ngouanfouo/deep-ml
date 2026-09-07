def kv_store(commands):
    """
    Processes a sequence of key-value store commands and returns the collected output as a single string.
    """
    current_time = 0
    store = {}  # key -> (value, expiry)
    backups = {}  # name -> {key: (value, remaining_ttl)}
    outputs = []
    
    def is_expired(expiry):
        if expiry is None:
            return False
        return current_time >= expiry

    for cmd in commands:
        op = cmd[0]
        
        if op == "TICK":
            current_time += int(cmd[1])
            
        elif op == "SET":
            key = cmd[1]
            val = cmd[2]
            if len(cmd) > 3:
                ttl = int(cmd[3])
                expiry = current_time + ttl
            else:
                expiry = None
            store[key] = (val, expiry)
            
        elif op == "GET":
            key = cmd[1]
            if key in store:
                if is_expired(store[key][1]):
                    del store[key]
                    outputs.append("NULL")
                else:
                    outputs.append(store[key][0])
            else:
                outputs.append("NULL")
                
        elif op == "DELETE":
            key = cmd[1]
            if key in store:
                if is_expired(store[key][1]):
                    del store[key]
                    outputs.append("false")
                else:
                    del store[key]
                    outputs.append("true")
            else:
                outputs.append("false")
                
        elif op == "SCAN":
            prefix = cmd[1]
            # Clean up expired keys
            expired_keys = [k for k, (_, exp) in store.items() if is_expired(exp)]
            for k in expired_keys:
                del store[k]
            
            matches = []
            for k, (val, exp) in store.items():
                if k.startswith(prefix):
                    matches.append((k, val))
            matches.sort(key=lambda x: x[0])
            
            if not matches:
                outputs.append("(empty)")
            else:
                formatted = ", ".join(f"{k}={val}" for k, val in matches)
                outputs.append(formatted)
                
        elif op == "BACKUP":
            name = cmd[1]
            # Clean up expired keys before snapshot
            expired_keys = [k for k, (_, exp) in store.items() if is_expired(exp)]
            for k in expired_keys:
                del store[k]
            
            snapshot = {}
            for k, (val, exp) in store.items():
                rem_ttl = None if exp is None else (exp - current_time)
                snapshot[k] = (val, rem_ttl)
            backups[name] = snapshot
            outputs.append(str(len(snapshot)))
            
        elif op == "RESTORE":
            name = cmd[1]
            store = {}
            if name in backups:
                for k, (val, rem_ttl) in backups[name].items():
                    expiry = None if rem_ttl is None else (current_time + rem_ttl)
                    store[k] = (val, expiry)
            outputs.append("OK")
            
    return "\n".join(outputs)