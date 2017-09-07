import random


class ElitistMutation:
    def __init__(self, mutation_rate, value_range, perturb_chance, perturb_range, elitist_cutoff):
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        self.perturb_chance = perturb_chance
        self.perturb_range = perturb_range
        self.cutoff = elitist_cutoff

    def execute(self, model, population):
        random.seed(None)
        flat_population = [c[0] for s in population for c in s]
        population_size = len(flat_population)
        num_mutations = int((population_size - self.cutoff) * flat_population[0].size * self.mutation_rate)

        mutation_points = random.sample(range(flat_population[0].size * self.cutoff,
                                              population_size * flat_population[0].size),
                                        num_mutations)

        for point in mutation_points:
            if random.uniform(0, 1) > self.perturb_chance:
                flat_population[point // flat_population[0].size][point % flat_population[0].size] = \
                    random.uniform(self.value_range[0], self.value_range[1])
            else:
                flat_population[point // flat_population[0].size][point % flat_population[0].size] += \
                    random.uniform(self.perturb_range[0], self.perturb_range[1])
