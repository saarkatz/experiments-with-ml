import random
import numpy as np


class MinPlayer:
    def __init__(self, name, cols, rows, seed=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.seed = seed
        self.action = np.zeros(self.cols)

    def init(self):
        random.seed(self.seed)

    def next_turn(self, state):
        direction = random.randint(0, 1)
        action = np.zeros(self.cols)
        min_col = (-1, 10)
        cols = range(self.cols)
        if direction:
            cols = reversed(cols)
        for i in cols:
            if state[i, 2] and not state[i, 3]:
                action[i] = 1
                return action
            j = 0
            while j < self.rows and state[i, j]:
                j += 1
            if min_col[1] > j:
                min_col = (i, j)
        action[min_col[0]] = 1
        return action
