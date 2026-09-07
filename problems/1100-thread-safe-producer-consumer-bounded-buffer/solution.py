import queue
import threading


def run_producer_consumer(capacity, producer_items, num_consumers):
    """
    Coordinates concurrent producers and consumers through a shared bounded buffer.

    Args:
        capacity (int): Maximum number of items the buffer can hold at once.
        producer_items (list of list): Items each producer thread will place into the buffer.
        num_consumers (int): Number of consumer threads removing items from the buffer.

    Returns:
        list: Sorted list of all consumed items.
    """
    q = queue.Queue(maxsize=capacity)
    consumed_items = []
    consumed_lock = threading.Lock()

    def producer_worker(items):
        for item in items:
            q.put(item)

    def consumer_worker():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            with consumed_lock:
                consumed_items.append(item)
            q.task_done()

    # Start consumer threads
    consumers = []
    for _ in range(num_consumers):
        t = threading.Thread(target=consumer_worker)
        t.start()
        consumers.append(t)

    # Start producer threads
    producers = []
    for items in producer_items:
        t = threading.Thread(target=producer_worker, args=(items,))
        t.start()
        producers.append(t)

    # Wait for all producers to finish pushing their items
    for t in producers:
        t.join()

    # Send termination sentinels (None) for each consumer
    for _ in range(num_consumers):
        q.put(None)

    # Wait for all consumers to finish processing
    for t in consumers:
        t.join()

    return sorted(consumed_items)