import random
import numpy as np


class RandomPlayer:
    def __init__(self, name, cols, rows, random_on_the_fly=False):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.random_on_the_fly = random_on_the_fly
        self.action = np.zeros(self.cols)
        self.decision_index = 0

        if not random_on_the_fly:
            self.decisions = [random.randint(0, self.action.size - 1) for i in range(cols * rows)]

    def init(self):
        self.decision_index = 0

    def next_turn(self, state):
        self.action -= self.action
        if self.random_on_the_fly:
            current = random.randint(0, self.action.size - 1)
        else:
            current = self.decisions[self.decision_index]
            self.decision_index += 1
        self.action[current] = 1
        return self.action
