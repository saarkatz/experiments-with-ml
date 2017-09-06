import random as random
import numpy as np

from NeuralNetwork import create_layer, ReLU, Identity, Sigmoid


class GAOptimizer:
    def __init__(self, population_size, fit_function, nn):
        self.population_size = population_size
        self.fit_function = fit_function
        self.population = []
        self.nn = nn
        for i in range(population_size):
            nn.init_model()
            self.population.append([np.nan, nn.get_weights_as_vector()])

        # For analysis
        self.generation = 0
        self.fittest_chromosome = []
        self.max_fit = []
        self.average_fit = []
        self.min_fit = []

    def survival_policy(self, extinction_rate):
        cut_off_index = int(len(self.population) * extinction_rate)
        self.population = self.population[:cut_off_index]
        return cut_off_index

    def select(self):
        pair = random.sample(range(len(self.population)), 2)
        return pair

    # chromosomes are 1D arrays that are the chosen chromosomes to crossover
    # points to cut are the number of points in the chromosome to cut in, if it's floating point between 0 and 1
    # then it indicates the probability of each segment to be cut (i.e 0.3 indicates 30% to cut at each point)
    def crossover(self, points_to_cut, first_chromosome, second_chromosome):
        new_chromosome = np.copy(first_chromosome)
        comp_chromosome = np.copy(second_chromosome)
        if points_to_cut >= 1:
            cutting_points = random.sample(range(1, new_chromosome.size - 1), points_to_cut)
            cutting_points.extend([0, new_chromosome.size - 1])
            cutting_points.sort()
            for first_index, second_index in zip(cutting_points[::2], cutting_points[1::2]):
                new_chromosome[first_index:second_index], comp_chromosome[first_index:second_index] = \
                    comp_chromosome[first_index:second_index], new_chromosome[first_index:second_index].copy()
        else:
            for k in range(0, new_chromosome.size):
                prob = random.randint(0, 1)
                if prob < points_to_cut:  # cut
                    new_chromosome[k:k+1], comp_chromosome[k:k+1] = \
                        comp_chromosome[k:k+1], new_chromosome[k:k+1].copy()
        return [np.nan, new_chromosome], [np.nan, comp_chromosome]

    def mutate(self, weights_range, percentage_of_mutations):
        # Split the range of the weights
        min_weight = weights_range[0]
        max_weight = weights_range[1]

        # Calculate the total number of mutations to do
        num_of_mutations = int(percentage_of_mutations * self.population_size * self.population[1][1].size)

        # Choose the chromosomes to mutate
        chromosomes_mutated = np.random.randint(1, self.population_size, size=num_of_mutations)

        # Count the number of mutations for each chromosome
        chromosomes_histogram = np.bincount(chromosomes_mutated)
        chromosomes_mutated = np.unique(chromosomes_mutated)

        # Mutate the chromosome
        for chromosome in chromosomes_mutated:
            num_of_occurrences = min(chromosomes_histogram[chromosome], int(self.population[0][1].size/2))
            curr_chromosome = self.population[chromosome][1]
            mutation_points = random.sample(range(curr_chromosome.size), num_of_occurrences)
            for point in mutation_points:
                curr_chromosome[point] = random.uniform(min_weight, max_weight)

    def breed(self, num_generations, extinction_rate, points_to_cut, weights_range, mutations_rate):
        for i in range(num_generations):
            # Evaluate the chromosomes
            for chromosome in self.population:
                self.nn.set_weights_from_vector(chromosome[1])
                chromosome[0] = self.fit_function(self.nn)
            self.population.sort(key=lambda x: x[0])

            # Record statistics
            self.generation += 1
            self.fittest_chromosome.append(self.population[-1][1].copy())
            self.max_fit.append(self.population[-1][0])
            self.min_fit.append(self.population[0][0])
            self.average_fit.append(np.mean([fit[0] for fit in self.population]))

            # Execute genetic operations
            num_new_chromosomes = self.survival_policy(extinction_rate)
            new_chromosomes = []
            for j in range(int(num_new_chromosomes/2)):
                mother_index, father_index = self.select()
                new_chromosomes.extend(self.crossover(points_to_cut,
                                                      self.population[mother_index][-1],
                                                      self.population[father_index][-1]))
            self.population.extend(new_chromosomes)
            self.mutate(weights_range, mutations_rate)


from Games.FourInARow import FourInARow
from Games.FourInARow.AiAgent import AiAgent
from Games.FourInARow.MinPlayer import MinPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Optimizers.TrainANetwork import test_net
import pickle

def fit_four_in_a_row(nn, max_games, random_turns, opponent):
    agent = AiAgent('Player', 7, 6, nn, False)
    multiple_opponents = isinstance(opponent, list)
    if multiple_opponents:
        opponents = opponent
    if multiple_opponents:
        game = FourInARow(7, 6, 4, None, agent, 500)
    else:
        game = FourInARow(7, 6, 4, opponent, agent, 500)
    sum = -1
    is_second = True
    lost = False
    while not lost and sum < max_games:
        if multiple_opponents:
            opponent = random.choice(opponents)
        if random_turns:
            is_second = np.random.randint(2)
        else:
            is_second = not is_second
        if is_second:
            game.player0, game.player1 = opponent, agent
        else:
            game.player0, game.player1 = agent, opponent
        game.init()
        result = game.run()
        reward = result[2] if result[0] == is_second else -result[2]
        lost = reward == -1
        sum += 1
    return sum


if __name__ == "__main__":
    x = create_layer(None, 7 * 6)
    w = create_layer(ReLU, 7 * 3, x, False)
    w2 = create_layer(ReLU, 7 * 2, w, False)
    nn = create_layer(Sigmoid, 7, w2, False)

    x_o = create_layer(None, 7 * 6)
    w = create_layer(ReLU, 7 * 3, x, False)
    w2 = create_layer(ReLU, 7 * 2, w, False)
    rand_opponent_nn = create_layer(Sigmoid, 7, w2, False)
    opponent = AiAgent('Rand', 7, 6, rand_opponent_nn, True)
    first_opponent = MinPlayer('player', 7, 6)
    const_opponent = ConstPlayer('const', 7, 6)

    def fit_function(x):
        if not opponent.use_prob:
            opponent.use_prob = False
            return fit_four_in_a_row(x, 100, False, opponent)
            opponent.use_prob = True
        else:
            return fit_four_in_a_row(x, 100, False, [first_opponent, const_opponent])

    gaOptimizer = GAOptimizer(100, fit_function, nn)

    # with open('gaOptimizer_save.pkl', 'rb') as file:
    #     pickle.load(gaOptimizer, file)

    while True:
        bench = 0.0
        for i in range(10):
            gaOptimizer.breed(10, 0.5, 0.1, [-1, 1], 0.001)
            print("Generation {0}:".format(gaOptimizer.generation))
            nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            if not opponent.use_prob:
                bench = test_net(nn, opponent)
            else:
                print(test_net(nn, const_opponent))
                bench = test_net(nn, first_opponent)
            print(bench)
            print("Max: " + str(gaOptimizer.max_fit[-1]))
            print("Average: " + str(gaOptimizer.average_fit[-1]))
            print("Min: " + str(gaOptimizer.min_fit[-1]))
        np.save('ga_net.npy', gaOptimizer.fittest_chromosome[-1])
        with open('gaOptimizer_save.pkl', 'wb') as file:
            pickle.dump(gaOptimizer, file)
        if bench > 0.9:
            print('Increasing hardness')
            opponent.nn.set_weights_from_vector(gaOptimizer.fittest_chromosome[-1])
            opponent.use_prob = False


