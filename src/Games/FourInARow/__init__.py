import time
import numpy as np


def mtime():
    return int(round(time.time() * 1000))


class FourInARow:
    def __init__(self, cols, rows, goal, player0, player1, timeout):
        self.cols = cols
        self.rows = rows
        self.goal = goal
        self.player0 = player0
        self.player1 = player1
        self.timeout = timeout
        self.state = None
        self.turn_queue = None
        self._initialized = 0

    def _init_state(self):
        self.state = np.zeros(shape=(self.cols, self.rows))

    def _in_col(self, index):
        total = 0
        while total < self.rows and not self.state[index, total] == 0:
            total += 1
        return total

    def _validate_action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)
        if not action.size == self.cols:
            return False
        else:
            action.reshape((self.cols,))
        if not sum(action) == 1:
            return False
        index = np.argmax(action)
        if not action[index] == 1:
            return False
        if not self._in_col(index) < self.rows:
            return False
        return True

    def _check_in_range(self, pos):
        if pos[0] < 0 or pos[0] >= self.cols or pos[1] < 0 or pos[1] >= self.rows:
            return False
        return True

    def _count(self, value, pos, direction, both_sides):
        if not self._check_in_range(pos):
            return 0
        if not self.state[tuple(pos)] == value:
            return 0
        else:
            if both_sides:
                return 1 + \
                       self._count(value, pos + direction, direction, False) + \
                       self._count(value, pos - direction, -direction, False)
            else:
                return 1 + self._count(value, pos + direction, direction, False)

    def _take_action(self, action):
        index = np.argmax(np.absolute(action))
        value = action[index]
        height = self._in_col(index)
        self.state[index, height] = value

    def _check_win_condition(self, action):
        index = np.argmax(action)
        height = self._in_col(index) - 1
        value = self.state[index, height]
        directions = [np.array((-1,-1)), np.array((0, -1)), np.array((1, -1)), np.array((1, 0))]
        for direction in directions:
            if self._count(value, np.array((index, height)), direction, both_sides=True) >= self.goal:
                return True
        return False

    def _run_turn(self, player, turn, fail_on_timeout):
        start = mtime()
        action = player.next_turn(self.state if turn % 2 == 0 else -self.state)
        end = mtime()
        elapsed = end - start
        self.turn_queue.append((turn, player.name, self.state.copy() if turn % 2 == 0 else -self.state.copy(), action))
        if not self._validate_action(action):
            return -1
        self._take_action(action if turn % 2 == 0 else -action)
        if elapsed > self.timeout:
            if fail_on_timeout:
                return -1
            else:
                print('Turn {0}: player {1} has timed out with {2} ms out of {3} ms'.format(turn, player.name, elapsed,
                                                                                            self.timeout))
        if self._check_win_condition(action):
            return 1
        return 0

    def init(self):
        self._init_state()
        self.turn_queue = []
        self.player0.init()
        self.player1.init()
        self._initialized = 1

    def run(self, fail_on_timeout=False):
        if not self._initialized == 1:
            return 0
        turn = 0
        while True:
            turn_result = self._run_turn(self.player0 if turn % 2 == 0 else self.player1, turn, fail_on_timeout)
            if not turn_result == 0:
                self._initialized = 2
                return (turn % 2, turn, turn_result)
            turn += 1


import pickle
from Games.FourInARow.NNAgent import NNAgent
from Games.FourInARow.PolicyAgent import PolicyAgent
from Games.FourInARow.HorizontalPlayer import HorizontalPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from Games.FourInARow.Player import Player
from NeuralNetwork import create_layer, ReLU, Sigmoid
from MonteCarloTree import MonteCarloTree


if __name__ == '__main__':
    # np.seterr(all='raise')
    # x = create_layer(None, 6 * 7)
    # w1 = create_layer(ReLU, 3 * 7, x, False)
    # w2 = create_layer(ReLU, 2 * 7, w1, False)
    # out = create_layer(Sigmoid, 7, w2)
    mct = MonteCarloTree(7, None, do_explore=False)

    with open('../../mcts_policy_7.pkl', 'rb') as file:
        mct.policy = pickle.load(file)

    # out.load('../../Optimizers/GAOptimizer2/ga_net_fit6.npy')

    # player0 = NNAgent('Player0', 7, 6, out, print_action=True)
    # player0 = PolicyAgent('Player0', 7, 6, mct)
    player0 = Player('Player0', 7, 6)
    i = 181
    player1 = PolicyAgent('Player1', 7, 6, mct)
    ge = FourInARow(7, 6, 4, player0, player1, 60000)
    while True:
        ge.init()
        result = ge.run()
        print(result)
        # print(ge.turn_queue)
        # with open('../../../game_history/game_{0}.pkl'.format(i), 'wb') as file:
        #     pickle.dump(ge.turn_queue, file)
        i += 1
    # 3345442122331561541522
