import random
import numpy as np


class MinPlayer:
    def __init__(self, name, cols, rows, random_on_the_fly=False):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.random_on_the_fly = random_on_the_fly
        self.action = np.zeros(self.cols)
        self.decision_index = 0

        if not random_on_the_fly:
            self.decisions = [random.randint(0, 1) for i in range(cols * rows)]

    def init(self):
        self.decision_index = 0

    def next_turn(self, state):
        if self.random_on_the_fly:
            direction = random.randint(0, 1)
        else:
            direction = self.decisions[self.decision_index]
            self.decision_index += 1
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
