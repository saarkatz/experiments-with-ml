import os
import numpy as np

from GameEngine import GameEngine
from AiAgent1 import AiAgent
from NeuralNetwork import create_placeholder, create_dense_layer


def train_net(num_games, callback=(lambda x, y, z: print(x) if x % 100 == 0 else None), opponent=None,
              source=None):
    x = create_placeholder('input', 6 * 7)
    w1 = create_dense_layer('w1', 3 * 7, x)
    w2 = create_dense_layer('w2', 50, w1)
    out = create_dense_layer('out', 7, w2)

    if source:
        out.load(source)

    # x = create_placeholder('input', 6 * 7)
    # w1 = create_dense_layer('w1', 50, x)
    # w2 = create_dense_layer('w2', 50, w1)
    # out_2 = create_dense_layer('out', 7, w2)

    player0 = AiAgent('Player0', 7, 6, out)
    if opponent:
        player1 = opponent
    else:
        player1 = player0

    engine = GameEngine(7, 6, 4, player0, player1, 500)

    for i in range(num_games):
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
            if is_second:
                callback(i + 1, result, player1)
            else:
                callback(i + 1, result, player0)

        # Get outcome
        reward = result[2] if result[0] == is_second else -result[2]

        # Get data set
        data_set = [(x[2].flatten(), x[3]) for x in turn_queue if x[0] % 2 == is_second]

        # Teach the network to take the good actions
        engine.player0.nn.learn(data_set, reward=reward)

    return player0


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


if __name__ == '__main__':
    ai_agent = train_net(20000, callback=lambda x, y, z: save_check_point(x, 'check_point', 1000, 100, y, z))
    ai_agent.nn.save('new_final_net.npy')
