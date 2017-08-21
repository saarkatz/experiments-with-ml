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

    # chromosomes are 1D arrays that are the chosen chromosomes to crossover
    # points to cut are the number of points in the chromosome to cut in, if it's between 0 and 1
    # then it indicates the probability of each segment to be cut
    def crossover(self, points_to_cut, first_chromosome, second_chromosome):
        new_chromosome = np.copy(first_chromosome)
        comp_chromosome = np.copy(second_chromosome)
        if points_to_cut >= 1:
            last_cutting_point = 0
            cutting_points = (random.sample(range(1, self.population_size), points_to_cut)).sort()
            for index in cutting_points:
                new_chromosome[last_cutting_point:index], comp_chromosome[last_cutting_point:index] = \
                    comp_chromosome[last_cutting_point:index], new_chromosome[last_cutting_point:index].copy()
                last_cutting_point = index
        else:
            for k in range(0, new_chromosome.size() - 2):
                prob = random.randint(0, 1)
                if prob < points_to_cut:  # cut
                    new_chromosome[k:k+1], comp_chromosome[k:k+1] = \
                        comp_chromosome[k:k+1], new_chromosome[k:k+1].copy()
        return new_chromosome, comp_chromosome

    def mutate(self, weights_range, percentage_of_mutations):
        min = weights_range[0]
        max = weights_range[1]
        num_of_mutations = percentage_of_mutations * self.population_size
        chromosomes_mutated = np.random.rand(num_of_mutations, 1)
        chromosomes_mutated = chromosomes_mutated * (self.population_size + 1)
        chromosomes_mutated = np.floor(chromosomes_mutated)
        chromsomes_histogram = np.bincount(chromosomes_mutated)
        chromosomes_mutated = np.unique(chromosomes_mutated)
        for k in range(chromosomes_mutated.size):
            num_of_occurences = chromsomes_histogram[chromosomes_mutated[k]]
            curr_chromosome = self.population[chromosomes_mutated[k]]
            if num_of_occurences > 0:
                genes_mutated = [random.randint(0, curr_chromosome) for x in range(num_of_occurences)]
                for gene in genes_mutated:
                    curr_chromosome[gene] = random.randint(min, max)








