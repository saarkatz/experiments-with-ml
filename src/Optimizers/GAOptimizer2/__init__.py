import random as random
import numpy as np


class GAOptimizer2:
    def __init__(self, ga_model, survival_policy, crossover_policy, mutation_policy, speciation_policy,
                 fitness_function):
        self.model = ga_model
        self.survival_policy = survival_policy
        self.crossover_policy = crossover_policy
        self.mutation_policy = mutation_policy
        self.speciation_policy = speciation_policy
        self.fitness_function = fitness_function

        # The population is composed by the species which in turn are composed of the genomes of
        # that species. This should be maintained by the speciation policy. The genomes are a pairs of
        # chromosome and fit value
        self.population = None

        # For analysis
        self.generation = 0
        self.fittest_chromosome = []
        self.max_fit = []
        self.average_fit = []
        self.min_fit = []

    def init(self, population_size):
        self.population = self.model.generate_pop(population_size)
        self.speciation_policy.execute(self.model, self.population)

    def iterate(self, num_generations):
        for i in range(num_generations):
            # Evaluate the chromosomes
            self.fitness_function.eval(self.model, self.population)

            # Record statistics
            self.generation += 1
            self.fittest_chromosome.append(self.fitness_function.best_chromosome(self.population))
            self.max_fit.append(self.fitness_function.max_fit(self.population))
            self.min_fit.append(self.fitness_function.min_fit(self.population))
            self.average_fit.append(self.fitness_function.average_fit(self.population))

            # Execute genetic operations
            total_population = self.survival_policy.execute(self.model, self.population)
            self.crossover_policy.execute(self.model, self.population, total_population)
            self.mutation_policy.execute(self.model, self.population)
            self.speciation_policy.execute(self.model, self.population)


from NeuralNetwork import create_layer, ReLU, Sigmoid

from Optimizers.GAOptimizer2.ElitistMutation import ElitistMutation
from Optimizers.GAOptimizer2.HighestPercentileSurvival import HighestPercentileSurvival
from Optimizers.GAOptimizer2.RandomPairsCrossover import RandomPairsCrossover
from Optimizers.GAOptimizer2.NoSpeciation import NoSpeciation
from Optimizers.GAOptimizer2.NNModel import NNModel

from Games.FourInARow import FourInARow
from Games.FourInARow.NNAgent import NNAgent
from Games.FourInARow.HorizontalPlayer import HorizontalPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.RandomPlayer import RandomPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from Optimizers.TrainANetwork import test_net
import pickle


def horizontal_setting(index, agent, before):
    if before:
        agent.use_random = False
        agent.current = index // 2 % 7
        agent.direction = index // 14
    else:
        agent.use_random = True


def const_setting(index, agent, before):
    if before:
        agent.use_random = False
        agent.current = index // 7
    else:
        agent.use_random = True


def random_setting(index, agent, before):
    if before:
        agent.seed = index
    else:
        agent.seed = None


def min_setting(index, agent, before):
    if before:
        agent.seed = index
    else:
        agent.seed = None


class FourInARowFitFunction:
    def __init__(self, games_cap=50):
        self.cap = games_cap

        self.opponent_list = []
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 7, random_setting))
        self.opponent_list.append((ConstPlayer('const', 7, 6), 14, const_setting))
        self.opponent_list.append((HorizontalPlayer('horizontal', 7, 6), 28, horizontal_setting))
        self.opponent_list.append((MinPlayer('min', 7, 6), 7, min_setting))

        self.game = FourInARow(7, 6, 4, None, None, 500)

    def add_opponent(self, new_opponent):
        self.opponent_list.append(new_opponent)
        if sum([o[1] for o in self.opponent_list]) > self.cap:
            self.opponent_list.pop(0)

    def _fit_function(self, agent):
        lost = False
        score = 0
        for opp in self.opponent_list:
            for i in range(opp[1]):
                if opp[2]:
                    opp[2](i, opp[0], True)
                if i % 2 == 0:
                    self.game.player0 = agent
                    self.game.player1 = opp[0]
                    is_second = False
                else:
                    self.game.player0 = opp[0]
                    self.game.player1 = agent
                    is_second = True
                self.game.init()
                result = self.game.run()
                reward = result[2] if result[0] == is_second else -result[2]
                lost = reward == -1
                if not lost:
                    score += 43 - result[1]
            if opp[2]:
                opp[2](None, opp[0], False)
        return score

    def eval(self, model, population):
        for species in population:
            for chromosome in species:
                chromosome[1] = self._fit_function(model.get_agent(chromosome[0]))
            species.sort(key=lambda x: x[1], reverse=True)

            # # Calculate shared score
            # average = sum((c[1] for c in species))/len(species)

            # # Set the score of the first chromosome (as it is the representative of the species) to the species score
            # species[0][1] = average
        # Sort species according to their score (In descending order)
        # population.sort(key=lambda x: x[0][1], reverse=True)

    def best_chromosome(self, population):
        return population[0][0][0]

    def max_fit(self, population):
        return population[0][0][1]

    def min_fit(self, population):
        return population[-1][-1][1]

    def average_fit(self, population):
        flat_population = [c[1] for s in population for c in s]
        return sum(flat_population) / len(flat_population)


if __name__ == "__main__":
    x = create_layer(None, 7 * 6)
    w = create_layer(Sigmoid, 7 * 6 * 4, x, True)
    w2 = create_layer(Sigmoid, 7 * 6 * 2, w, True)
    w3 = create_layer(Sigmoid, 7 * 6 * 1, w2, True)
    nn = create_layer(Sigmoid, 7, w3, False)

    ff = FourInARowFitFunction()

    gaOptimizer = GAOptimizer2(NNModel(NNAgent('Player', 7, 6, nn)),
                               HighestPercentileSurvival(0.5),
                               RandomPairsCrossover(2, 0, True),
                               ElitistMutation(0.05, [-1, 1], 0.9, [-0.01, 0.01], 2),
                               NoSpeciation(),
                               ff)

    gaOptimizer.init(24)

    # with open('gaOptimizer2_fit7_lnn2_save.pkl', 'rb') as file:
    #     gaOptimizer = pickle.load(file)
    #     gaOptimizer.fitness_function = ff

    while True:
        bench = 0.0
        for i in range(5):
            gaOptimizer.iterate(20)
            print("Generation {0}:".format(gaOptimizer.generation))
            print("Max: " + str(gaOptimizer.max_fit[-1]))
            print("Average: " + str(gaOptimizer.average_fit[-1]))
            print("Min: " + str(gaOptimizer.min_fit[-1]))
            nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            bench = test_net(nn, ff.opponent_list[0][0])
            print('Test vs opp.{0}: {1}'.format(0, bench))
            for j, opponent in zip(range(1,4), ff.opponent_list[1:4]):
                print('Test vs opp.{0}: {1}'.format(j, test_net(nn, opponent[0])))
        if bench > 0.9:
            print('Increasing hardness')
            x = create_layer(None, 7 * 6)
            w = create_layer(Sigmoid, 7 * 6 * 4, x, True)
            w2 = create_layer(Sigmoid, 7 * 6 * 2, w, True)
            w3 = create_layer(Sigmoid, 7 * 6 * 1, w2, True)
            o = create_layer(Sigmoid, 7, w3, False)
            opponent = NNAgent('Opponent', 7, 6, o)
            opponent.nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            ff.add_opponent((opponent, 2, None))
        np.save('ga_net_fit8_lnn4.npy', gaOptimizer.fittest_chromosome[-1])
        with open('gaOptimizer2_fit8_lnn4_save.pkl', 'wb') as file:
            pickle.dump(gaOptimizer, file)

