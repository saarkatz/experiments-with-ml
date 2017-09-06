import numpy as np

class HighestPercentileSurvival:
    def __init__(self, cutoff_percentile):
        self.cutoff = cutoff_percentile

    def execute(self, model, population):
        total_population = sum((len(s) for s in population))
        surviving_count = int(total_population * self.cutoff)
        # median = np.median(np.unique((f[1] for s in population for f in s)))

        new_population = []
        running_population_count = 0
        for s in population:
            if running_population_count + len(s) >= surviving_count:
                left_over = surviving_count - running_population_count
                new_population.append(s[:left_over])
                running_population_count += len(new_population[-1])
                break
            else:
                running_population_count += len(s)
                new_population.append(s)
        population[:] = new_population

        return total_population
