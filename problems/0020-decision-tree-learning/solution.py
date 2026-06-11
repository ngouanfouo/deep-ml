import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    if not labels:
        return 0.0
    
    label_counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    
    for count in label_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Calculate entropy of the parent node
    parent_labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(parent_labels)
    
    # Calculate weighted average entropy of children
    # Group examples by attribute value
    attr_values = {}
    for example in examples:
        value = example[attr]
        if value not in attr_values:
            attr_values[value] = []
        attr_values[value].append(example)
    
    # Calculate weighted entropy
    total_examples = len(examples)
    weighted_entropy = 0.0
    
    for value, subset in attr_values.items():
        subset_labels = [example[target_attr] for example in subset]
        subset_entropy = calculate_entropy(subset_labels)
        weight = len(subset) / total_examples
        weighted_entropy += weight * subset_entropy
    
    # Information gain = parent_entropy - weighted_entropy
    return parent_entropy - weighted_entropy


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    if not examples:
        return None
    
    labels = [example[target_attr] for example in examples]
    label_counts = Counter(labels)
    
    # Find max count
    max_count = max(label_counts.values())
    
    # Get all labels with max count
    most_common_labels = [label for label, count in label_counts.items() if count == max_count]
    
    # Sort alphabetically and return the first
    most_common_labels.sort()
    
    return most_common_labels[0]


def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # Base case 1: No examples left
    if not examples:
        return None
    
    # Base case 2: All examples have the same class
    labels = [example[target_attr] for example in examples]
    if len(set(labels)) == 1:
        return labels[0]
    
    # Base case 3: No attributes left to split on
    if not attributes:
        return majority_class(examples, target_attr)
    
    # Choose the best attribute to split on
    best_attr = None
    best_gain = -1
    
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        # Tie-breaking: choose the first attribute in the list if gains are equal
        # Using > (not >=) ensures first attribute is chosen when gains are equal
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    
    # If no attribute gives positive gain, return majority class
    if best_attr is None or best_gain <= 0:
        return majority_class(examples, target_attr)
    
    # Create the tree node
    tree = {best_attr: {}}
    
    # Get remaining attributes (remove the chosen one)
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    
    # Get all possible values for the best attribute
    attr_values = sorted(set(example[best_attr] for example in examples))
    
    # Split examples by attribute value and recurse
    for value in attr_values:
        # Get subset of examples with this attribute value
        subset = [example for example in examples if example[best_attr] == value]
        
        # Recursively learn subtree
        subtree = learn_decision_tree(subset, remaining_attrs, target_attr)
        
        # Add branch to tree
        tree[best_attr][value] = subtree
    
    return tree