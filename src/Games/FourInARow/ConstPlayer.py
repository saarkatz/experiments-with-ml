import random
import numpy as np


class ConstPlayer:
    def __init__(self, name, cols, rows, use_random=True, seed=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.use_random = use_random
        self.seed = seed
        self.action = np.zeros(self.cols)
        self.current = 0

    def init(self):
        if self.use_random:
            random.seed(self.seed)
            self.current = random.randint(0, self.cols - 1)
        self.action -= self.action
        self.action[self.current] = 1

    def next_turn(self, state):
        if not state[np.argmax(self.action), -1] == 0:
            self.init()
        return self.action
