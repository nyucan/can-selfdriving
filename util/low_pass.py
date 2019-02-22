from collections import deque

class LowPass(object):
    def __init__(self, num=5):
        self.q = deque()
        self.max_size = num
        self.s = 0

    def apply(self, newEle):
        if self.max_size == len(self.q):
            self.s -= self.q.popleft()
        self.s += newEle
        self.q.append(newEle)
        return self.s / len(self.q)
