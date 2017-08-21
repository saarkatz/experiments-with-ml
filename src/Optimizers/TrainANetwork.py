import os

import numpy as np

from Games.FourInARow import FourInARow
from Games.FourInARow.AiAgent1 import AiAgent
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from NeuralNetwork import create_placeholder, create_dense_layer


def train_net(nn, num_games, learn_rate=1e-3, gamma=0.99, lambda_reg=0.5,
              callback=(lambda x, y, z: print(x) if x % 100 == 0 else None), opponent=None):
    player0 = AiAgent('Player0', 7, 6, nn)
    multiple_opponents = isinstance(opponent, list)
    if opponent:
        if not multiple_opponents:
            player1 = opponent
        else:
            player1 = None
    else:
        player1 = player0

    engine = FourInARow(7, 6, 4, player0, player1, 500)

    for i in range(num_games):
        if multiple_opponents:
            # Randomize the opponent
            player1 = np.random.choice(opponent)

        # Randomize the players
        is_second = np.random.randint(0, 2)
        if is_second:
            engine.player0, engine.player1 = player1, player0
        else:
            engine.player0, engine.player1 = player0, player1

        # Initialize the game
        engine.init()

        # Run the game
        result = engine.run()

        # Get game history
        turn_queue = engine.turn_queue

        # Call callback
        if callback:
            callback(i + 1, result, player0)

        # Get outcome
        reward = result[2] if result[0] == is_second else -result[2]
        if reward < 0:
            if result[2] < 0:
                gamma *= 1e-2
            else:
                reward *= 1e-2

        # Get data set
        data_set = [(x[2].flatten(), x[3]) for x in turn_queue if x[0] % 2 == is_second]

        # Calculate backward reward
        running_reward = reward
        back_reward = [running_reward]
        for _ in range(len(data_set) - 1):
            back_reward.append(back_reward[-1] * gamma)

        # Teach the network to take the good actions
        for pair, r in zip(data_set, reversed(back_reward)):
            player0.nn.learn([pair], learn_rate=learn_rate, reward=r, lambda_reg=lambda_reg)

    return nn


def echo_partial_results(step, directory, game_index, game_result, engine):
    if game_index % step == 0:
        with open(os.path.join(directory, 'Game_{0}_result.txt'.format(game_index))) as file:
            file.write(engine.turn_queue)
        engine.player0.nn.save(os.path.join(directory, 'Ai_{0}_result'.format(game_index)))


def save_check_point(step, file_name, save_rate, message_rate, game_result, ai_agent):
    if message_rate and step % message_rate == 0:
        print(step)
    if save_rate and step % save_rate == 0:
        ai_agent.nn.save(file_name)


def test_net(nn, opponent):
    engine = FourInARow(7, 6, 4, None, None, 60000)
    player0 = AiAgent('nn', 7, 6, nn)

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
    print(wins/games)


if __name__ == '__main__':
    x = create_placeholder('input', 6 * 7)
    w1 = create_dense_layer('w1', 3 * 7, x, has_bias=False)
    w2 = create_dense_layer('w2', 2 * 7, w1, has_bias=False)
    out = create_dense_layer('out', 7, w2)

    x = create_placeholder('input', 6 * 7)
    opponent_nn = create_dense_layer('out', 7, x)
    rand_opponent = AiAgent('rand_opponent', 7, 6, opponent_nn, use_prob=True)
    const_opponent = ConstPlayer('const_opponent', 7, 6)
    opponent = MinPlayer('opponent', 7, 6)

    # out.load('net_1_1.npy')

    test_net(out, opponent)
    test_net(out, rand_opponent)
    test_net(out, const_opponent)

    count = 0
    iterations = 20000
    while True:
        # opponent.nn.set_weights_from_vector(out.get_weights_as_vector())
        train_net(out, iterations, learn_rate=1e-4, gamma=0.99, lambda_reg=1,
                  callback=lambda x, y, z: save_check_point(x + count * iterations, 'check_point', 1000, 1000, y, z),
                  opponent=opponent)  # [opponent, rand_opponent, const_opponent])
        out.save('net_strong_a3_{0}.npy'.format(count))
        test_net(out, opponent)
        test_net(out, rand_opponent)
        test_net(out, const_opponent)
        count += 1
