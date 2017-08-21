from NeuralNetwork import NeuralNetwork, init_model
import random as random
import  numpy as np
class GAOptimizer:
    def __init__(self, population_size, fit_function, nn):
        self.population_size = population_size
        self.fit_function = fit_function
        self.population = []
        for i in range(population_size):
            nn.init_model()
            temp_vector = nn.get_weights_as_vector()
            self.population.append(temp_vector)

    def crossover(self, points_to_cut, first_chromosome, second_chromosome):
        new_chromosome = np.copy(first_chromosome)
        comp_chromosome = np.copy(second_chromosome)
        if points_to_cut >= 1:
            last_cutting_point = 0
            cutting_points = (random.sample(range(1, len(self.population)), points_to_cut)).sort()
            for index in cutting_points:
                new_chromosome[last_cutting_point:index], comp_chromosome[last_cutting_point:index] = \
                    comp_chromosome[last_cutting_point:index], new_chromosome[last_cutting_point:index].copy()
                last_cutting_point = index
        else:
            for k in range(0, new_chromosome.size() - 2):
                x = random.randint(0, 1)
                if x < points_to_cut:  # cut
                    new_chromosome[k:k+1], comp_chromosome[k:k+1] = \
                        comp_chromosome[k:k+1], new_chromosome[k:k+1].copy()
        return new_chromosome, comp_chromosome




