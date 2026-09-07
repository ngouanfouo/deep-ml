import heapq


def process_transactions(commands):
    """
    Processes a chronological list of banking commands and returns the outputs
    produced by query commands (balance and top) in order.
    """
    balances = {}
    scheduled = []  # Min-heap of (exec_time, reg_id, src, dst, amount)
    reg_counter = 0
    outputs = []
    
    def process_scheduled(current_time):
        while scheduled and scheduled[0][0] <= current_time:
            _, _, src, dst, amount = heapq.heappop(scheduled)
            src_bal = balances.get(src, 0)
            if src_bal >= amount:
                balances[src] = src_bal - amount
                balances[dst] = balances.get(dst, 0) + amount

    for cmd in commands:
        op = cmd[0]
        time = cmd[1]
        
        # Process any pending scheduled transfers due by this command's timestamp
        process_scheduled(time)
        
        if op == "deposit":
            account = cmd[2]
            amount = cmd[3]
            balances[account] = balances.get(account, 0) + amount
            
        elif op == "transfer":
            src = cmd[2]
            dst = cmd[3]
            amount = cmd[4]
            src_bal = balances.get(src, 0)
            if src_bal >= amount:
                balances[src] = src_bal - amount
                balances[dst] = balances.get(dst, 0) + amount
                
        elif op == "schedule":
            src = cmd[2]
            dst = cmd[3]
            amount = cmd[4]
            exec_time = cmd[5]
            if src not in balances:
                balances[src] = 0
            if dst not in balances:
                balances[dst] = 0
            heapq.heappush(scheduled, (exec_time, reg_counter, src, dst, amount))
            reg_counter += 1
            
        elif op == "balance":
            account = cmd[2]
            outputs.append(balances.get(account, 0))
            if account not in balances:
                balances[account] = 0
                
        elif op == "top":
            k = cmd[2]
            sorted_accounts = sorted(balances.keys(), key=lambda acc: (-balances[acc], acc))
            outputs.append(sorted_accounts[:k])
            
    return outputs