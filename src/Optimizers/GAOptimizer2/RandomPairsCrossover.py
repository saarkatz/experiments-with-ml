import numpy as np
import random


class RandomPairsCrossover:
    def __init__(self, num_cuts, inter_species_crossover_rate, use_swap):
        self.num_cuts = num_cuts
        self.inter_species_crossover_rate = inter_species_crossover_rate
        self.use_swap = use_swap

    def execute(self, model, population, total_population):
        new_gourp = []
        current_population = sum((len(s) for s in population))
        missing_population = total_population - current_population
        for _ in range(missing_population//2):
            if len(population) < 2 or random.uniform(0, 1) > self.inter_species_crossover_rate:
                species = random.sample(population, 1)[0]
                if len(species) < 2:
                    pair = (species[0], species[0])
                else:
                    pair = random.sample(species, 2)
            else:
                species = random.sample(population, 2)
                pair = [s[0] for s in species]

            new_chromosome = np.copy(pair[0][0])
            comp_chromosome = np.copy(pair[1][0])

            cutting_points = random.sample(range(1, new_chromosome.size - 1), self.num_cuts)
            cutting_points.extend([0, new_chromosome.size - 1])
            cutting_points.sort()
            if self.use_swap:
                for first_index, second_index in zip(cutting_points[::2], cutting_points[1::2]):
                    new_chromosome[first_index:second_index], comp_chromosome[first_index:second_index] = \
                        comp_chromosome[first_index:second_index], new_chromosome[first_index:second_index].copy()
            else:
                for first_index, second_index in zip(cutting_points[::2], cutting_points[1::2]):
                    new_segment = (comp_chromosome[first_index:second_index] +
                                   new_chromosome[first_index:second_index])/2
                    new_chromosome[first_index:second_index] = new_segment
                    comp_chromosome[first_index:second_index] = new_segment

            new_gourp.append([new_chromosome, 0])
            new_gourp.append([comp_chromosome, 0])

        if missing_population%2 == 1:
            if random.uniform(0, 1) > self.inter_species_crossover_rate:
                species = random.sample(population, 1)[0]
                if len(species) < 2:
                    pair = (species[0], species[0])
                else:
                    pair = random.sample(species, 2)
            else:
                species = random.sample(population, 2)
                pair = [s[0] for s in species]

            new_chromosome = np.copy(pair[0][0])
            comp_chromosome = np.copy(pair[1][0])

            cutting_points = random.sample(range(1, new_chromosome.size - 1), self.num_cuts)
            cutting_points.extend([0, new_chromosome.size - 1])
            cutting_points.sort()

            if self.use_swap:
                for first_index, second_index in zip(cutting_points[::2], cutting_points[1::2]):
                    new_chromosome[first_index:second_index] = comp_chromosome[first_index:second_index]
            else:
                for first_index, second_index in zip(cutting_points[::2], cutting_points[1::2]):
                    new_segment = (comp_chromosome[first_index:second_index] +
                                   new_chromosome[first_index:second_index])/2
                    new_chromosome[first_index:second_index] = new_segment

            new_gourp.append([new_chromosome, 0])

        population.append(new_gourp)