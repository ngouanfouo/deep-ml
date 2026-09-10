def parse_inventory(s):
    totals = {}
    for record in s.split(';'):
        record = record.strip()
        if not record:
            continue                # skip empty records
        
        if ':' in record:
            name, qty_str = record.split(':', 1)
            name = name.strip()
            qty_str = qty_str.strip()
            qty = 0 if qty_str == '' else int(qty_str)
        else:
            name = record.strip()
            qty = 0
        
        if not name:
            continue                # skip records with empty name
        
        totals[name] = totals.get(name, 0) + qty
    
    return [f"{name}={totals[name]}" for name in sorted(totals)]