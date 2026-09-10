from collections import Counter

def count_common_unique(list1, list2):
    # list1: list of strings
    # list2: list of strings
    # return an integer
    c1 = Counter(list1)
    c2 = Counter(list2)
    return sum(1 for w, cnt in c1.items() if cnt == 1 and c2.get(w, 0) == 1)