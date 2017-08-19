import numpy as np


class AiAgent:
    def __init__(self, name, cols, rows, nn, source=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.nn = nn
        if source:
            nn.load(source)

    @staticmethod
    def _boltzmann_value(x, a, cut_off=0.1):
        result = np.zeros(x.shape)
        mask = x > cut_off * a
        partial_x = x[mask]
        result[mask] = np.exp(-a/partial_x)
        return result

    @staticmethod
    def _raw_to_prob(action):
        a = AiAgent._boltzmann_value(action, 0.5)
        a = a/sum(a)
        return a

    def next_turn(self, state):
        if not isinstance(state, np.ndarray):
            state = np.asarray(state)
        state_vector = state.flatten()
        action_out = self.nn.run({'input': state_vector})
        action_prob = AiAgent._raw_to_prob(action_out)
        action = np.zeros(action_out.shape)
        action[np.random.choice(len(action_out), p=action_prob)] = 1
        print(repr(action_out))
        print(repr(action_prob))
        return action
