class NNModel:
    def __init__(self, nn_agent):
        self.agent = nn_agent

    def get_agent(self, chromosome):
        self.agent.nn.set_weights_from_vector(chromosome)
        return self.agent

    def generate_pop(self, population_size):
        population = [[]]
        for _ in range(population_size):
            self.agent.nn.init_model()
            population[0].append([self.agent.nn.get_weights_as_vector(), 0])
        return population