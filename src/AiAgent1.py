import numpy as np

class AiAgent:
    def __init__(self, name, cols, rows, nn, source=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.nn = nn
        if source:
            nn.load(source)

    def next_turn(self, state):
        if not isinstance(state, np.ndarray):
            state = np.asarray(state)
        state_vector = state.flatten()
        action_out = self.nn.run({'input': state_vector})
        action = np.zeros(action_out.shape)
        action[np.argmax(action_out)] = 1
        return action
