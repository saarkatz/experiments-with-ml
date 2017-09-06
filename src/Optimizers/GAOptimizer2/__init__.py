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


from NeuralNetwork import create_layer, ReLU, Identity, Sigmoid

from Optimizers.GAOptimizer2.ElitistMutation import ElitistMutation
from Optimizers.GAOptimizer2.HighestPercentileSurvival import HighestPercentileSurvival
from Optimizers.GAOptimizer2.RandomPairsCrossover import RandomPairsCrossover
from Optimizers.GAOptimizer2.NoSpeciation import NoSpeciation
from Optimizers.GAOptimizer2.NNModel import NNModel

from Games.FourInARow import FourInARow
from Games.FourInARow.AiAgent import AiAgent
from Games.FourInARow.RandomPlayer import RandomPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Optimizers.TrainANetwork import test_net
import pickle


class FourInARowFitFunction:
    def __init__(self):
        self.initial_opponents = 3
        self.opponent_list = []
        self.opponent_list.append(ConstPlayer('const', 7, 6))
        self.opponent_list.append(RandomPlayer('rand', 7, 6))
        self.opponent_list.append(MinPlayer('min', 7, 6))
        self.game = FourInARow(7, 6, 4, None, None, 500)

    def add_opponent(self, new_opponent):
        self.opponent_list.pop(0)
        self.opponent_list.append(new_opponent)
        if self.initial_opponents:
            self.initial_opponents -= 1

    def _fit_function(self, agent):
        lost = False
        score = 0
        while not lost:
            if score == 60:
                break
            if self.initial_opponents and score < 20 * self.initial_opponents:
                self.opponent_list[score // 20].seed = score
            if score % 2 == 0:
                self.game.player0 = agent
                self.game.player1 = self.opponent_list[score // 20]
                is_second = False
            else:
                self.game.player0 = self.opponent_list[score // 20]
                self.game.player1 = agent
                is_second = True
            self.game.init()
            result = self.game.run()
            reward = result[2] if result[0] == is_second else -result[2]
            lost = reward == -1
            if not lost:
                score += 1
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
        population.sort(key=lambda x: x[0][1], reverse=True)

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

    gaOptimizer.init(24)

    # with open('gaOptimizer2_save.pkl', 'rb') as file:
    #     gaOptimizer = pickle.load(file)
    #     ff = gaOptimizer.fitness_function

    while True:
        bench = 0.0
        for i in range(5):
            gaOptimizer.iterate(20)
            print("Generation {0}:".format(gaOptimizer.generation))
            print("Max: " + str(gaOptimizer.max_fit[-1]))
            print("Average: " + str(gaOptimizer.average_fit[-1]))
            print("Min: " + str(gaOptimizer.min_fit[-1]))
            nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            for j, opponent in enumerate(ff.opponent_list[:-1]):
                print('Test vs opp.{0}: {1}'.format(j, test_net(nn, opponent)))
            bench = test_net(nn, ff.opponent_list[-1])
            print('Test vs opp.2: {0}'.format(bench))
        np.save('ga_net.npy', gaOptimizer.fittest_chromosome[-1])
        with open('gaOptimizer2_save.pkl', 'wb') as file:
            pickle.dump(gaOptimizer, file)
        if bench > 0.9:
            print('Increasing hardness')
            x = create_layer(None, 7 * 6)
            w = create_layer(ReLU, 7 * 3, x, False)
            w2 = create_layer(ReLU, 7 * 2, w, False)
            o = create_layer(Sigmoid, 7, w2, False)
            opponent = AiAgent('Opponent', 7, 6, o)
            opponent.nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            ff.add_opponent(opponent)

