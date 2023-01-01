import heapq


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        cur_item = None
        for i in self.heap:
            if i[2] == item:
                cur_item = i

        if not cur_item:
            self.push(item, priority)
        elif priority < cur_item[0]:
            self.heap.remove(cur_item)
            c= (priority,cur_item[1],cur_item[2])
            self.heap.append(c)
            heapq.heapify(self.heap)


p = PriorityQueue()

p.push('a',1)

p.push('b',2)
p.push('c',3)
p.push('d',4)
p.push('e',2)


# p.update('a',3)
# p.update('c',3)

# print(p.heap)
# print("\n> updating prioroty of 'a' to 3 :")
# p.update('c',1)
print(p.heap)
print(p.pop())
print(p.pop())
print(p.pop())
print(p.pop())
print(p.pop())


