import random
import numpy as np


class ConstPlayer:
    def __init__(self, name, cols, rows, use_random=True):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.use_random = use_random
        self.action = np.zeros(self.cols)
        self.current = 3

    def init(self):
        if self.use_random:
            self.current = random.randint(0, self.cols - 1)

    def next_turn(self, state):
        i = 0
        while i < self.cols and not state[self.current, -1] == 0:
            self.current = (self.current + 1) % self.cols
            i += 1
        self.action -= self.action
        self.action[self.current] = 1

        return self.action
