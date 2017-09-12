import random
import numpy as np


# Wrapper AI that improves upon lesser AIs
class Mark2:
    def __init__(self, name, cols, rows, goal, player):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.goal = goal
        self.rollout_player = player

    def init(self):
        self.rollout_player.init()

    def _in_col(self, state, index):
        total = 0
        while total < self.rows and not state[index, total] == 0:
            total += 1
        return total

    def _validate_action(self, state, action):
        index = np.argmax(action)
        if not state[index, -1] == 0:
            return False
        return True

    def _check_in_range(self, pos):
        if pos[0] < 0 or pos[0] >= self.cols or pos[1] < 0 or pos[1] >= self.rows:
            return False
        return True

    def _count(self, state, value, pos, direction, both_sides, phantom):
        if not self._check_in_range(pos):
            return 0
        if not phantom and not state[tuple(pos)] == value:
            return 0
        else:
            if both_sides:
                return 1 + \
                       self._count(state, value, pos + direction, direction, False, False) + \
                       self._count(state, value, pos - direction, -direction, False, False)
            else:
                return 1 + self._count(state, value, pos + direction, direction, False, False)

    def _check_win_condition(self, state, action):
        index = np.argmax(np.abs(action))
        height = self._in_col(state, index)
        value = action[index]
        directions = [np.array((-1,-1)), np.array((0, -1)), np.array((1, -1)), np.array((1, 0))]
        for direction in directions:
            if self._count(state, value, np.array((index, height)), direction, True, True) >= self.goal:
                return True
        return False

    def next_turn(self, state):
        # If there is a winning move, take it
        opponent_win = False
        opponent_action = None
        for action in np.eye(self.cols):
            if self._validate_action(state, action):
                if self._check_win_condition(state, action):
                    return action
                if not opponent_win and self._check_win_condition(state, -action):
                    opponent_action = action
                    opponent_win = True
        if opponent_win:
            return opponent_action

        action = self.rollout_player.next_turn(state)
        # If the move will result in a foul and there is another option, take it
        current = np.argmax(action)
        i = 0
        while i < self.cols and not state[current, -1] == 0:
            current = (current + 1) % self.cols
            i += 1
        action -= action
        action[current] = 1
        return action
