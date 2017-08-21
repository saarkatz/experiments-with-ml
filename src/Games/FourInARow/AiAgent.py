import numpy as np


class AiAgent:
    def __init__(self, name, cols, rows, nn, use_prob=False, print_action=False, source=None):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.nn = nn
        self.use_prob = use_prob
        self.print_action = print_action
        if source:
            nn.load(source)

    def init(self):
        pass

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
        action_out = self.nn.run(state_vector)
        action = np.zeros(action_out.shape)
        if self.print_action:
            print(repr(action_out))
        if self.use_prob:
            action_prob = AiAgent._raw_to_prob(action_out)
            action[np.random.choice(len(action_out), p=action_prob)] = 1
            if self.print_action:
                print(repr(action_prob))
        action[np.argmax(action_out)] = 1
        return action
