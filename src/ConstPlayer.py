import numpy as np


class ConstPlayer:
    def __init__(self, name, cols, rows):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.action = np.zeros(self.cols)

    def init(self):
        self.action -= self.action
        self.action[np.random.randint(0, self.cols)] = 1

    def next_turn(self, state):
        if not state[np.argmax(self.action), -1] == 0:
            self.init()
        return self.action