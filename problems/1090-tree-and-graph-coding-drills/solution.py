def tree_and_graph_drills(task, *args):
    if task == "lca":
        tree, root, p, q = args
        
        def lca(node):
            if node is None:
                return None
            if node == p or node == q:
                return node
            left = lca(tree[node][0]) if tree[node][0] is not None else None
            right = lca(tree[node][1]) if tree[node][1] is not None else None
            if left is not None and right is not None:
                return node
            return left if left is not None else right
        
        return lca(root)
    
    elif task == "clone":
        graph = args[0]
        cloned = {}
        def dfs(node):
            if node in cloned:
                return cloned[node]
            new_node = node  # keys are the same values
            cloned[node] = []
            for nb in graph[node]:
                cloned[node].append(nb)
                dfs(nb)
            return cloned[node]
        for node in graph:
            dfs(node)
        # Sort
        return {k: sorted(cloned[k]) for k in sorted(cloned)}
    
    elif task == "min_removals":
        s = args[0]
        balance = 0
        unmatched_close = 0
        for ch in s:
            if ch == '(':
                balance += 1
            else:
                if balance > 0:
                    balance -= 1
                else:
                    unmatched_close += 1
        return balance + unmatched_close
    
    elif task == "word_break":
        s, words = args
        word_set = set(words)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[n]