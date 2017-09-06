import random
import numpy as np


class RandomPlayer:
    def __init__(self, name, cols, rows, seed=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.seed = seed
        self.action = np.zeros(self.cols)

    def init(self):
        random.seed(self.seed)

    def next_turn(self, state):
        self.action -= self.action
        self.action[random.randint(0, self.action.size - 1)] = 1
        return self.action
