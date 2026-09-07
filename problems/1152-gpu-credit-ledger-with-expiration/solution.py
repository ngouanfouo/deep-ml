def ledger(operations):
    """
    Processes a billing ledger of GPU compute credits with support for out-of-order
    timestamps and expiration rules.

    Args:
        operations (list of tuples): Sequence of 'add', 'charge', and 'balance' operations.

    Returns:
        list: Results for every 'charge' and 'balance' operation in order.
    """
    grants = []
    outputs = []
    
    for op in operations:
        tag = op[0]
        
        if tag == "add":
            _, credits, expiry = op
            grants.append({
                'id': len(grants),
                'credits': float(credits) if isinstance(credits, float) else int(credits),
                'expiry': expiry
            })
            
        elif tag == "charge":
            _, amount, time = op
            # Find valid grants: expiry > time and credits > 0
            valid_grants = [g for g in grants if g['expiry'] > time and g['credits'] > 0]
            # Sort by expiry ascending, then by arrival id ascending
            valid_grants.sort(key=lambda g: (g['expiry'], g['id']))
            
            consumed = 0
            for g in valid_grants:
                if amount <= 0:
                    break
                take = min(amount, g['credits'])
                g['credits'] -= take
                consumed += take
                amount -= take
                
            outputs.append(consumed)
            
        elif tag == "balance":
            _, time = op
            # Sum unspent credits from grants valid at the given time (expiry > time)
            total_balance = sum(g['credits'] for g in grants if g['expiry'] > time and g['credits'] > 0)
            outputs.append(total_balance)
            
    return outputs