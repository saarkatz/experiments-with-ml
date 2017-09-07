import random
import numpy as np


class HorizontalPlayer:
    def __init__(self, name, cols, rows, use_random=True, seed=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.use_random = use_random
        self.seed = seed
        self.action = np.zeros(self.cols)
        self.direction = 0
        self.current = 0

    def init(self):
        if self.use_random:
            random.seed(self.seed)
            self.current = random.randint(0, self.cols - 1)
            self.direction = random.randint(0, 1) * 2 - 1

    def next_turn(self, state):
        direction = random.randint(0, 1)
        action = np.zeros(self.cols)
        action[self.current] = 1
        self.current = (self.current + self.direction) % self.cols
        return action
