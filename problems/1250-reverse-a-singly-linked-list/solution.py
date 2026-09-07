class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(values, method="iterative"):
    """
    Reverses a singly linked list given its values in head-to-tail order,
    using either an 'iterative' or 'recursive' approach.

    Args:
        values (list): Node values in head-to-tail order.
        method (str): 'iterative' or 'recursive'.

    Returns:
        list: Node values of the reversed linked list in head-to-tail order.
    """
    if not values:
        return []

    # Step 1: Build the linked list from the input values
    head = ListNode(values[0])
    curr = head
    for val in values[1:]:
        curr.next = ListNode(val)
        curr = curr.next

    # Step 2: Reverse the linked list based on the chosen method
    if method == "iterative":
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        head = prev
    elif method == "recursive":
        def reverse_recursive(curr, prev):
            if not curr:
                return prev
            next_node = curr.next
            curr.next = prev
            return reverse_recursive(next_node, curr)

        head = reverse_recursive(head, None)

    # Step 3: Extract the values from the reversed linked list
    result = []
    curr = head
    while curr:
        result.append(curr.val)
        curr = curr.next

    return result