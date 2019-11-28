class Node:
    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    def connect(self, other):
        self.next = other
        other.prev = self

    # Insert node between self and self.next
    def splice(self, node):
        next = self.next
        self.connect(node)
        node.connect(next)

    def delete(self):
        self.prev.connect(self.next)


class LinkedList:
    def __init__(self):
        self.front = Node()
        self.back = Node()
        self.front.connect(self.back)

    def push_back(self, node):
        self.back.prev.splice(node)

    def pop_front(self):
        if self.front.next == self.back:
            print("Empty Linked List!\n")
            return
        res = self.front.next  # self.front.next is the first node with data
        self.front.next.delete()
        return res

    def delete_node(self, node):
        if self.front.next == self.back:
            print("Empty Linked List!\n")
            return
        node.delete()


# ll = LinkedList()
# ll.push_back(1)
# print(ll.front.next.data)
# print(ll.back)