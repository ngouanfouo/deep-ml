import heapq


def order_book(operations):
    """
    Simulates a limit order book supporting insert, cancel, best_bid, and best_ask operations.

    Args:
        operations (list of tuples): Sequence of operations to process.

    Returns:
        list: Results for each operation in order.
    """
    bids_heap = []  # Max-heap stored as negative prices: (-price, timestamp, order_id)
    asks_heap = []  # Min-heap: (price, timestamp, order_id)
    resting_orders = {}  # order_id -> {'side', 'price', 'qty', 'timestamp'}
    
    timestamp = 0
    results = []

    for op in operations:
        tag = op[0]

        if tag == "insert":
            _, order_id, side, price, quantity = op
            timestamp += 1
            trades = []

            if side == "buy":
                # Match against resting sell orders (asks_heap) where ask price <= buy price
                while quantity > 0:
                    while asks_heap:
                        p, ts, oid = asks_heap[0]
                        if oid not in resting_orders or resting_orders[oid]['qty'] <= 0:
                            heapq.heappop(asks_heap)
                        else:
                            break

                    if not asks_heap:
                        break

                    p, ts, oid = asks_heap[0]
                    if p > price:
                        break

                    ask_order = resting_orders[oid]
                    trade_qty = min(quantity, ask_order['qty'])
                    # [incoming_order_id, resting_order_id, trade_price, traded_quantity]
                    trades.append([order_id, oid, p, trade_qty])

                    quantity -= trade_qty
                    ask_order['qty'] -= trade_qty

                    if ask_order['qty'] == 0:
                        del resting_orders[oid]
                        heapq.heappop(asks_heap)

                if quantity > 0:
                    resting_orders[order_id] = {
                        'side': 'buy',
                        'price': price,
                        'qty': quantity,
                        'timestamp': timestamp,
                    }
                    heapq.heappush(bids_heap, (-price, timestamp, order_id))

            elif side == "sell":
                # Match against resting buy orders (bids_heap) where bid price >= sell price
                while quantity > 0:
                    while bids_heap:
                        neg_p, ts, oid = bids_heap[0]
                        if oid not in resting_orders or resting_orders[oid]['qty'] <= 0:
                            heapq.heappop(bids_heap)
                        else:
                            break

                    if not bids_heap:
                        break

                    neg_p, ts, oid = bids_heap[0]
                    p = -neg_p
                    if p < price:
                        break

                    bid_order = resting_orders[oid]
                    trade_qty = min(quantity, bid_order['qty'])
                    trades.append([order_id, oid, p, trade_qty])

                    quantity -= trade_qty
                    bid_order['qty'] -= trade_qty

                    if bid_order['qty'] == 0:
                        del resting_orders[oid]
                        heapq.heappop(bids_heap)

                if quantity > 0:
                    resting_orders[order_id] = {
                        'side': 'sell',
                        'price': price,
                        'qty': quantity,
                        'timestamp': timestamp,
                    }
                    heapq.heappush(asks_heap, (price, timestamp, order_id))

            results.append(trades)

        elif tag == "cancel":
            _, order_id = op
            if order_id in resting_orders:
                del resting_orders[order_id]
                results.append(True)
            else:
                results.append(False)

        elif tag == "best_bid":
            while bids_heap:
                neg_p, ts, oid = bids_heap[0]
                if oid not in resting_orders or resting_orders[oid]['qty'] <= 0:
                    heapq.heappop(bids_heap)
                else:
                    break
            if not bids_heap:
                results.append(None)
            else:
                results.append(-bids_heap[0][0])

        elif tag == "best_ask":
            while asks_heap:
                p, ts, oid = asks_heap[0]
                if oid not in resting_orders or resting_orders[oid]['qty'] <= 0:
                    heapq.heappop(asks_heap)
                else:
                    break
            if not asks_heap:
                results.append(None)
            else:
                results.append(asks_heap[0][0])

    return results