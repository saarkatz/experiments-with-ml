import os
import numpy as np

from GameEngine import GameEngine
from AiAgent1 import AiAgent
from NeuralNetwork import create_placeholder, create_dense_layer


def train_new_net(num_games, callback=(lambda x, y, z: print(x) if x % 100 == 0 else None)):
    x = create_placeholder('input', 6 * 7)
    w1 = create_dense_layer('w1', 50, x)
    w2 = create_dense_layer('w2', 50, w1)
    out = create_dense_layer('out', 7, w2)

    # x = create_placeholder('input', 6 * 7)
    # w1 = create_dense_layer('w1', 50, x)
    # w2 = create_dense_layer('w2', 50, w1)
    # out_2 = create_dense_layer('out', 7, w2)

    player0 = AiAgent('Player0', 7, 6, out)
    player1 = AiAgent('Player1', 7, 6, out)

    engine = GameEngine(7, 6, 4, player0, player1, 500)

    for i in range(num_games):
        # Randomize the players
        if np.random.randint(0, 2):
            engine.player0, engine.player1 = engine.player1, engine.player0
        engine.init()
        result = engine.run()
        turn_queue = engine.turn_queue
        if callback:
            callback(i, result, engine)
        winner = result[0] if result[2] == 1 else 1 - result[0]
        winner_ai = engine.player1 if winner else engine.player0
        # looser_ai = engine.player0 if winner else engine.player1
        w_dataset = [(x[2].flatten(), x[3]) for x in turn_queue if x[0] % 2 == winner]
        # l_dataset = [(x[2].flatten(), 1 - x[3]) for x in turn_queue if not x[0] % 2 == winner]
        winner_ai.nn.learn(w_dataset, iterations=1)
        # for input_vec, output_vec in l_dataset:
        #     looser_ai.nn.learn(input_vec, output_vec, iterations=1)

    return engine


def echo_partial_results(step, directory, game_index, game_result, engine):
    if game_index % step == 0:
        with open(os.path.join(directory, 'Game_{0}_result.txt'.format(game_index))) as file:
            file.write(engine.turn_queue)
        engine.player0.nn.save(os.path.join(directory, 'Ai_{0}_result'.format(game_index)))


if __name__ == '__main__':
    engine = train_new_net(500)
    engine.player0.nn.save('final_net_4_r')
    # engine.player1.nn.save('final_net_2')
