import numpy as np

class Player:
    def __init__(self, name, cols, rows):
        self.name = name
        self.cols = cols
        self.rows = rows

    def init(self):
        pass

    def next_turn(self, state):
        print(np.flipud(np.transpose(state)))
        input_action = int(input('Player %s: Enter your move:' % self.name))
        action = np.zeros((self.cols,))
        action[input_action] = 1
        return action
