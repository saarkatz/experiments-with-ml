import numpy as np


class WeightSpeciation:
    def __init__(self, weight_coefficient, thresh_hold):
        self.coefficient = weight_coefficient
        self.thresh_hold = thresh_hold

    def _distance(self, chromosome1, chromosome2):
        diff_vec = chromosome1 - chromosome2
        diff = np.mean(np.abs(diff_vec))
        return diff

    def execute(self, model, population):
        flat_population = [c for s in population for c in s]
        new_population = [[flat_population[0]]]
        for c in flat_population[1:]:
            added = False
            for s in new_population:
                dist = self._distance(s[0][0], c[0]) * self.coefficient
                if dist < self.thresh_hold:
                    s.append(c)
                    added = True
                    break
            if not added:
                new_population.append([c])
        population[:] = new_population
