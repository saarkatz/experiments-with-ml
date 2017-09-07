import numpy as np


class MonteCarloTree:
    def __init__(self, action_size, rollout_policy, do_explore=True):
        self.policy = {}
        self.action_size = action_size
        self.rollout_policy = rollout_policy
        self.do_explore = do_explore

    def record(self, state_action_reward):
        is_list = isinstance(state_action_reward, list)
        if is_list:
            for state, action, reward in state_action_reward:
                if state in self.policy:
                    self.policy[state][action][0] += reward
                    self.policy[state][action][1] += 1
                else:
                    self.policy[state] = [[0, 1] for i in range(self.action_size)]
                    self.policy[state][action][0] += reward
        else:
            if state_action_reward[0] in self.policy:
                self.policy[state_action_reward[0]][state_action_reward[1]][0] += state_action_reward[2]
                self.policy[state_action_reward[0]][state_action_reward[1]][1] += 1
            else:
                self.policy[state_action_reward[0]] = [[0, 1] for i in range(self.action_size)]
                self.policy[state_action_reward[0]][state_action_reward[1]][0] += state_action_reward[2]

    def get_action(self, state):
        if state in self.policy:
            w = np.array([a[0] for a in self.policy[state]])
            n = np.array([a[1] for a in self.policy[state]])
            if self.do_explore:
                s = w/n + 1.414 * np.sqrt(np.log(sum(n))/n)
            else:
                s = w/n
            action = np.zeros(self.action_size)
            action[np.argmax(s)] = 1
            return action
        else:
            if self.rollout_policy:
                return self.rollout_policy.get_action(state)
            else:
                action = np.zeros(self.action_size)
                action[np.random.randint(self.action_size)] = 1
                return action



import pickle
from Games.FourInARow import FourInARow
from Games.FourInARow.PolicyAgent import PolicyAgent, board_to_number
from Games.FourInARow.HorizontalPlayer import HorizontalPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.RandomPlayer import RandomPlayer
from Games.FourInARow.MinPlayer import MinPlayer


def test_agent(agent, opponent):
    engine = FourInARow(7, 6, 4, None, None, 60000)
    player0 = agent

    wins = 0
    games = 1000
    for i in range(games):
        is_second = np.random.randint(0, 2)
        if is_second:
            engine.player0, engine.player1 = opponent, player0
        else:
            engine.player0, engine.player1 = player0, opponent

        engine.init()
        result = engine.run()
        reward = result[2] if result[0] == is_second else -result[2]
        if reward > 0:
            wins += 1
    return wins/games


class AlwaysMiddlePolicy:
    def get_action(self, state):
        return np.array([0, 0, 0, 1, 0, 0, 0])


if __name__ == '__main__':
    mct_player = MonteCarloTree(7, None)
    mct = MonteCarloTree(7, None)

    # with open('mcts_policy_6.pkl', 'rb') as file:
    #     mct.policy = pickle.load(file)

    pa = PolicyAgent('policy', 7, 6, mct_player)
    game = FourInARow(7, 6, 4, pa, pa, 500)
    print('Training')

    # Learn from recorded games
    # for j in range(1, 201):
    #     game.init()
    #     with open('../game_history/game_{0}.pkl'.format(j), 'rb') as file:
    #         turn_queue = pickle.load(file)
    #     game.state = turn_queue[-1][2] if turn_queue[-1][0] % 2 == 0 else -turn_queue[-1][2]
    #     reward = int(game._validate_action(turn_queue[-1][3]))
    #     first_player_won = reward if not turn_queue[-1][0] % 2 else 1 - reward
    #     training_set = [(board_to_number(a[2]), np.argmax(a[3]),
    #                      first_player_won if a[0] % 2 == 0 else 1 - first_player_won)
    #                     for a in turn_queue]
    #     mct.record(training_set)

    i = 0
    num_iterations = 10000
    min_iterations = 10000
    while True:
        for _ in range(num_iterations):
            game.init()
            result = game.run()
            first_player_won = ((result[2] if not result[0] else -result[2]) + 1)//2
            training_set = [(board_to_number(a[2]), np.argmax(a[3]),
                             first_player_won if a[0] % 2 == 0 else 1 - first_player_won)
                            for a in game.turn_queue]
            mct_player.record(training_set)
        # mct_player.policy = mct.policy
        with open('mcts_policy_7.pkl', 'wb') as file:
            pickle.dump(mct_player.policy, file)
        i += num_iterations
        if num_iterations > min_iterations:
            num_iterations //= 10
        print('{0} games played'.format(i))
        # print('Test vs opp.horizontal: {0}'.format(test_agent(pa, HorizontalPlayer('horizontal', 7, 6))))
        # print('Test vs opp.const: {0}'.format(test_agent(pa, ConstPlayer('const', 7, 6))))
        # print('Test vs opp.rand: {0}'.format(test_agent(pa, RandomPlayer('rand', 7, 6))))
        # print('Test vs opp.min: {0}'.format(test_agent(pa, MinPlayer('min', 7, 6))))


