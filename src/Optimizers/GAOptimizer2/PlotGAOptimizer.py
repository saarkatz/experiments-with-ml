import matplotlib.pyplot as plt
import pickle

from NeuralNetwork import create_layer, ReLU, Sigmoid

from Optimizers.GAOptimizer2 import GAOptimizer2
from Optimizers.GAOptimizer2 import FourInARowFitFunction
from Optimizers.GAOptimizer2.ElitistMutation import ElitistMutation
from Optimizers.GAOptimizer2.HighestPercentileSurvival import HighestPercentileSurvival
from Optimizers.GAOptimizer2.RandomPairsCrossover import RandomPairsCrossover
from Optimizers.GAOptimizer2.NoSpeciation import NoSpeciation
from Optimizers.GAOptimizer2.NNModel import NNModel
from Optimizers.GAOptimizer2 import horizontal_setting
from Optimizers.GAOptimizer2 import const_setting
from Optimizers.GAOptimizer2 import random_setting
from Optimizers.GAOptimizer2 import min_setting

from Games.FourInARow import FourInARow
from Games.FourInARow.NNAgent import AiAgent
from Games.FourInARow.HorizontalPlayer import HorizontalPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.RandomPlayer import RandomPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from Optimizers.TrainANetwork import test_net


if __name__ == '__main__':
    x = create_layer(None, 7 * 6)
    w = create_layer(ReLU, 7 * 3, x, False)
    w2 = create_layer(ReLU, 7 * 2, w, False)
    nn = create_layer(Sigmoid, 7, w2, False)

    ff = FourInARowFitFunction()

    gaOptimizer = GAOptimizer2(NNModel(AiAgent('Player', 7, 6, nn)),
                               HighestPercentileSurvival(0.5),
                               RandomPairsCrossover(2, 0, True),
                               ElitistMutation(0.05, [-1, 1], 0.9, [-0.01, 0.01], 2),
                               NoSpeciation(),
                               ff)

    with open('gaOptimizer2_fit6_save.pkl', 'rb') as file:
        gaOptimizer = pickle.load(file)
        plt.plot(gaOptimizer.max_fit)
        plt.plot(gaOptimizer.average_fit)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.show()