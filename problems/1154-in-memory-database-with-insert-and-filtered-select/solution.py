def run_db(operations):
    """
    Executes a sequence of create, insert, and select operations on an in-memory database.

    Args:
        operations (list of tuples): Database commands to execute.

    Returns:
        list: Results of every select operation in order.
    """
    tables = {}
    select_results = []
    
    for op in operations:
        cmd = op[0]
        
        if cmd == "create":
            _, name, schema = op
            tables[name] = {
                'schema': schema,
                'col_map': {col: i for i, col in enumerate(schema)},
                'rows': []
            }
            
        elif cmd == "insert":
            _, name, values = op
            tables[name]['rows'].append(list(values))
            
        elif cmd == "select":
            # ("select", table_name, columns, where) or ("select", table_name, columns, where, order_by)
            name = op[1]
            columns = op[2]
            where = op[3]
            order_by = op[4] if len(op) > 4 else None
            
            table = tables[name]
            schema = table['schema']
            col_map = table['col_map']
            rows = table['rows']
            
            def evaluate_predicate(pred, row):
                if pred is None:
                    return True
                
                p_type = pred[0]
                if p_type == "and":
                    return evaluate_predicate(pred[1], row) and evaluate_predicate(pred[2], row)
                elif p_type == "or":
                    return evaluate_predicate(pred[1], row) or evaluate_predicate(pred[2], row)
                else:
                    # Leaf predicate: (op, column, value)
                    op_sign, col, val = pred
                    row_val = row[col_map[col]]
                    if op_sign == "=":
                        return row_val == val
                    elif op_sign == "!=":
                        return row_val != val
                    elif op_sign == "<":
                        return row_val < val
                    elif op_sign == ">":
                        return row_val > val
                    elif op_sign == "<=":
                        return row_val <= val
                    elif op_sign == ">=":
                        return row_val >= val
                return True

            # Step 1: Filter rows using the where predicate
            filtered_rows = [r for r in rows if evaluate_predicate(where, r)]
            
            # Step 2: Sort rows if order_by is specified (Python's sort is stable)
            if order_by is not None:
                col_name, direction = order_by
                col_idx = col_map[col_name]
                reverse = (direction == "desc")
                filtered_rows = sorted(filtered_rows, key=lambda r: r[col_idx], reverse=reverse)
                
            # Step 3: Project the requested columns
            if columns == ["*"]:
                proj_indices = list(range(len(schema)))
            else:
                proj_indices = [col_map[c] for c in columns]
                
            projected = [[r[i] for i in proj_indices] for r in filtered_rows]
            select_results.append(projected)
            
    return select_results