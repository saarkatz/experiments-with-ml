import random
import numpy as np


class ConstPlayer:
    def __init__(self, name, cols, rows, seed=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.seed = seed
        self.action = np.zeros(self.cols)

    def init(self):
        random.seed(self.seed)
        self.action -= self.action
        self.action[random.randint(0, self.cols - 1)] = 1

    def next_turn(self, state):
        if not state[np.argmax(self.action), -1] == 0:
            self.init()
        return self.action
