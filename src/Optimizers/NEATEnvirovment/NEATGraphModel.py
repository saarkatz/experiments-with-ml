import numpy as np

from Games.FourInARow.PolicyAgent import PolicyAgent


class NEATGraphModel:
    def __init__(self, graph):
        self.agent = graph

    def get_agent(self):
        return PolicyAgent('Neat', 7, 6, self)

    def get_action(self, state):
        raw_action = self.agent.evaluate_stable(np.concatenate(([1], state.flatten())), 0.1)
        action = np.zeros_like(raw_action)
        action[np.argmax(raw_action)] = 1
        return action

