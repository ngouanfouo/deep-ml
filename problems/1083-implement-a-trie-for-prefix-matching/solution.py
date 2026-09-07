class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        curr = self.root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return curr.is_end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True


def trie_operations(words, queries):
    """
    Builds a Trie from a list of words and processes a sequence of search/startsWith queries.
    
    Args:
        words (list of str): Strings to insert into the Trie.
        queries (list of tuple): Each query is a pair (op, value) where op is 'search' or 'startsWith'.
        
    Returns:
        list of bool: Results of each query in order.
    """
    trie = Trie()
    for word in words:
        trie.insert(word)
        
    results = []
    for op, value in queries:
        if op == "search":
            results.append(trie.search(value))
        elif op == "startsWith":
            results.append(trie.startsWith(value))
        else:
            results.append(False)
            
    return results