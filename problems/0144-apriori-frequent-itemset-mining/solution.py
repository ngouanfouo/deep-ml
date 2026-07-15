from collections import defaultdict
from itertools import combinations

def apriori(transactions, min_support, max_length=None):
    """
    Implement the Apriori algorithm to find frequent itemsets.
    
    Args:
        transactions: List of sets, each set contains items in a transaction
        min_support: Minimum support threshold (fractional, between 0 and 1)
        max_length: Maximum size of itemsets to consider (optional)
    
    Returns:
        Dictionary mapping frozenset of items to support (fractional)
    """
    # Handle empty transactions
    if not transactions:
        raise ValueError("Transactions list cannot be empty")
    
    # Validate min_support
    if not (0 <= min_support <= 1):
        raise ValueError("min_support must be between 0 and 1")
    
    num_transactions = len(transactions)
    min_count = min_support * num_transactions
    
    # Dictionary to store all frequent itemsets
    frequent_itemsets = {}
    
    # Step 1: Find frequent 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    # Filter 1-itemsets that meet minimum support
    frequent_1_itemsets = {
        frozenset([item]): count / num_transactions
        for item, count in item_counts.items()
        if count >= min_count
    }
    
    frequent_itemsets.update(frequent_1_itemsets)
    
    # If no frequent 1-itemsets, return empty
    if not frequent_1_itemsets:
        return frequent_itemsets
    
    # If max_length is 1, return just the 1-itemsets
    if max_length == 1:
        return frequent_itemsets
    
    # Step 2: Generate frequent k-itemsets for k >= 2
    current_level_itemsets = list(frequent_1_itemsets.keys())
    k = 2
    
    while current_level_itemsets:
        # Check if we've reached max_length
        if max_length is not None and k > max_length:
            break
        
        # Generate candidate k-itemsets from frequent (k-1)-itemsets
        candidates = set()
        
        # Join step: combine itemsets that share first k-2 items
        for i in range(len(current_level_itemsets)):
            for j in range(i + 1, len(current_level_itemsets)):
                itemset1 = sorted(current_level_itemsets[i])
                itemset2 = sorted(current_level_itemsets[j])
                
                # If first k-2 items are the same, create candidate
                if itemset1[:-1] == itemset2[:-1]:
                    candidate = frozenset(itemset1 + [itemset2[-1]])
                    candidates.add(candidate)
        
        # Prune step: remove candidates that have infrequent (k-1)-subsets
        pruned_candidates = []
        for candidate in candidates:
            # Check all (k-1)-subsets of candidate
            is_valid = True
            candidate_list = sorted(candidate)
            for subset in combinations(candidate_list, k - 1):
                if frozenset(subset) not in frequent_itemsets:
                    is_valid = False
                    break
            if is_valid:
                pruned_candidates.append(candidate)
        
        if not pruned_candidates:
            break
        
        # Count support for each candidate
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in pruned_candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        # Filter candidates that meet minimum support
        frequent_k_itemsets = {}
        for itemset, count in candidate_counts.items():
            if count >= min_count:
                support = count / num_transactions
                frequent_k_itemsets[itemset] = support
        
        # If no frequent k-itemsets, break
        if not frequent_k_itemsets:
            break
        
        # Add to frequent itemsets
        frequent_itemsets.update(frequent_k_itemsets)
        
        # Prepare for next iteration
        current_level_itemsets = list(frequent_k_itemsets.keys())
        k += 1
    
    return frequent_itemsets