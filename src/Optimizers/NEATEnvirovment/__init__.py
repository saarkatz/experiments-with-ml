from random import sample
from numpy import random
import numpy as np
import pickle

from Optimizers.NEATEnvirovment.NEATGraphModel import NEATGraphModel
from Games.FourInARow.RandomPlayer import RandomPlayer
from Games.FourInARow.ConstPlayer import ConstPlayer
from Games.FourInARow.HorizontalPlayer import HorizontalPlayer
from Games.FourInARow.MinPlayer import MinPlayer
from Games.FourInARow import FourInARow

def sigmoid(x, a):
    return 1/(1 + np.exp(-a * x))


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


class FourInARowFitFunction:
    instance = None

    def __init__(self, games_cap=50):
        self.cap = games_cap

        self.opponent_list = []
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((RandomPlayer('rand', 7, 6), 1, None))
        self.opponent_list.append((ConstPlayer('const', 7, 6), 14, const_setting))
        self.opponent_list.append((HorizontalPlayer('horizontal', 7, 6), 28, horizontal_setting))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))
        self.opponent_list.append((MinPlayer('min', 7, 6), 1, None))

        self.game = FourInARow(7, 6, 4, None, None, 500)

    def add_opponent(self, new_opponent):
        self.opponent_list.append(new_opponent)
        if sum([o[1] for o in self.opponent_list]) > self.cap:
            self.opponent_list.pop(0)

    def fit_function(self, agent):
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
                result = self.game.run(fail_on_timeout=True)
                reward = result[2] if result[0] == is_second else -result[2]
                lost = reward == -1
                if not lost:
                    score += 43 - result[1]
                else:
                    score += result[1]/43
            if opp[2]:
                opp[2](None, opp[0], False)
        return score
FourInARowFitFunction.instance = FourInARowFitFunction()


class NEATGraph:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.abj_matrix = None
        self.hidden_state = np.zeros(hidden_size)

    @classmethod
    def from_genome(cls, genome):
        graph = cls(genome.sensors, genome.outputs, len(genome.hidden))
        graph.update_abj_matrix(genome)
        return graph

    @classmethod
    def from_string(cls, string):
        string = string.split('|')
        input_size = int(string[0])
        abj_matrix = eval(string[1])
        hidden_size = abj_matrix.shape[0] - input_size
        output_size = abj_matrix.shape[1] - hidden_size
        graph = cls(input_size, output_size, hidden_size)
        graph.abj_matrix = abj_matrix
        return graph

    def update_abj_matrix(self, genome):
        # Create map from node id to index
        nodes_map = {g: i for i, g in enumerate(genome.hidden)}
        self.abj_matrix = np.zeros((self.input_size + self.hidden_size, self.output_size + self.hidden_size))
        for connection in (g for g in genome.connections if not g['dis']):
            from_to = []
            if connection['from'] < 0:
                from_to.append(connection['from'] + self.input_size)
            else:
                from_to.append(nodes_map[connection['from']] + self.input_size)
            if connection['to'] < 0:
                from_to.append(connection['to'] + self.input_size + self.output_size)
            else:
                from_to.append(nodes_map[connection['to']] + self.output_size)
            self.abj_matrix[tuple(from_to)] += connection['weight']

    # Return the output after n timesteps
    def timestep(self, input, n):
        for _ in range(n):
            state = np.concatenate((input, self.hidden_state))
            result = np.dot(state, self.abj_matrix)
            result = sigmoid(result, 4.9)  # TODO: This 4.9 here is from the NEAT paper.
            self.hidden_state = result[self.output_size:]
        return result[:self.output_size]

    def evaluate_stable(self, input, threshold, cap=100):
        self.hidden_state = np.zeros_like(self.hidden_state)
        state = np.concatenate((input, self.hidden_state))
        result = np.dot(state, self.abj_matrix)
        result = sigmoid(result, 4.9)
        self.hidden_state = result[self.output_size:]
        prev_result = np.zeros_like(result)
        i = 0
        while i < cap and max(np.abs((result - prev_result))) > threshold:
            i += 1
            prev_result = result
            state = np.concatenate((input, self.hidden_state))
            result = np.dot(state, self.abj_matrix)
            result = sigmoid(result, 4.9)
            self.hidden_state = result[self.output_size:]
        return result[:self.output_size]

    def evaluate_fixed(self, input, timesteps):
        self.hidden_state = np.zeros_like(self.hidden_state)
        return self.timestep(input, timesteps)

    def to_string(self):
        return str(self.input_size) + '|\nnp.' + repr(self.abj_matrix)


class NEATGenome:
    def __init__(self, species, sensors, outputs):
        self.species = species
        # The sensors and outputs will be counted as negative numbers (as if they where at the end and in reverse)
        self.sensors = sensors
        self.outputs = outputs
        self.hidden = []  # Node: innovation_num
        self.connections = []  # Conn: {in, out, weight, is_disabled, innovation_num} (in and out are innov num)
        self.fitness = 0
        self.real_fitness = 0

    @classmethod
    def reproduce(cls, mother_genome, father_genome):
        baby_genome = NEATGenome(mother_genome.species, mother_genome.sensors, mother_genome.outputs)
        # i_mother = 0
        # len_mother = len(mother_genome.hidden)
        # i_father = 0
        # len_father = len(father_genome.hidden)
        # while i_mother < len_mother or i_father < len_father:
        #     if i_mother < len_mother and i_father < len_father:
        #         equals = mother_genome.hidden[i_mother] == father_genome.hidden[i_father]
        #         mother_less_than_father = mother_genome.hidden[i_mother] < father_genome.hidden[i_father]
        #         if mother_less_than_father:
        #             baby_genome.hidden.append(mother_genome.hidden[i_mother])
        #             i_mother += 1
        #         else:
        #             baby_genome.hidden.append(father_genome.hidden[i_father])
        #             i_father += 1
        #         if equals:
        #             i_mother += 1
        #     elif i_mother < len_mother:
        #         baby_genome.hidden.append(mother_genome.hidden[i_mother])
        #         i_mother += 1
        #     else:
        #         baby_genome.hidden.append(father_genome.hidden[i_father])
        #         i_father += 1
        return baby_genome

    # Returns a copy of the gene
    def copy(self):
        new_genome = NEATGenome(self.species, self.sensors, self.outputs)
        new_genome.hidden = list(self.hidden)
        new_genome.connections = [gene.copy() for gene in self.connections]
        new_genome.fitness = self.fitness
        new_genome.real_fitness = self.real_fitness
        return new_genome

    # Returns the phenotype of the Genome in a functional form.
    def genesis(self):
        return NEATGraph.from_genome(self)

    # Returns true if the node is a connected node in genome
    def is_connected(self, node):
        for gene in self.connections:
            if gene['to'] == node or gene['from'] == node:
                return True
        return False


class NEATSpecies:
    def __init__(self, environment):
        self.id = environment.species_id
        environment.species_id += 1
        self.fitness = 0
        self.prev_fitness = 0
        self.unchanged_time = 0
        self.population = []
        self.representative = None

    def best_genome(self):
        return self.population[0]

    def update_parameters(self):
        self.representative = random.choice(self.population).copy()

        # TODO: Consider using the real fitness of the top member as the indication for stagnation.
        if self.population[0].real_fitness == self.prev_fitness:
            self.unchanged_time += 1
        else:
            self.unchanged_time = 0
        self.prev_fitness = self.population[0].real_fitness

    def add_genome(self, genome):
        self.population.append(genome)
        if not self.representative:
            self.representative = genome.copy()


class NEATEnvironment:
    # Information that needs to be known by the environment of the NEAT.
    def __init__(self, population_size, input_size, output_size, initial_size_factor, min_init_size_factor,
                 filter_percentage, stagnation_time, bolster_percentage, bolster_reduce_rate, minimum_bolster,
                 interspecies_crossover_chance,
                 elitist_threshold, mutate_connections_chance, new_connection_chance, new_node_chance, new_input_chance,
                 weight_perturb_chance, weight_soft_range, weight_perturb_range,
                 compatibility_threshold, disjoint_coeff, excess_coeff, weights_coeff,
                 target_species_number, threshold_adjustment_param):
        # Parameters for initialization
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.initial_size_factor = initial_size_factor
        self.min_init_size_factor = min_init_size_factor

        # Parameters for population filtering
        self.filter_percentage = filter_percentage
        self.stagnation_time = stagnation_time
        # TODO: Another options to implement the bolster is could be Geometric mean or Harmonic mean
        self.bolster_percentage = bolster_percentage
        self.bolster_reduce_rate = bolster_reduce_rate
        self.minimum_bolster = minimum_bolster

        # Parameters for crossover
        self.interspecies_crossover_chance = interspecies_crossover_chance

        # Parameters for mutations
        self.elitist_threshold = elitist_threshold
        self.mutate_connections_chance = mutate_connections_chance
        self.new_connection_chance = new_connection_chance
        self.new_node_chance = new_node_chance
        self.new_input_chance = new_input_chance
        self.weight_perturb_chance = weight_perturb_chance
        self.weight_soft_range = weight_soft_range
        self.weight_perturb_range = weight_perturb_range

        # Parameters for speciation
        self.compatibility_threshold = compatibility_threshold
        self.disjoint_coeff = disjoint_coeff
        self.excess_coeff = excess_coeff
        self.weights_coeff = weights_coeff
        self.target_species_number = target_species_number
        self.threshold_adjustment_param = threshold_adjustment_param

        # Members of the environment
        self.species_id = 0
        self.species = []
        self.mutations_list = {}
        self.connection_innovation = 0
        self.node_innovation = 0

        # History tracking
        self.generation = 0
        self.history = []
        self.species_history = []
        self.best_genomes = []
        self.species_fit = []
        self.fit_history = {'max': [], 'avg': [], 'min': []}

        # Initialize the environment
        self.init()

    def init(self):
        species = NEATSpecies(self)
        for i in range(self.population_size):
            genome = NEATGenome(species, self.input_size, self.output_size)
            for j in range(random.randint(self.input_size * self.output_size * self.min_init_size_factor,
                                          self.input_size * self.output_size * self.initial_size_factor)):
                if not self.mutate_add_connection(genome, unique=True):
                    print('Failed to add unique connection')
            species.add_genome(genome)
        self.species.append(species)
        self.speciate_population()
        self.fit_population()
        self.record_history()
        self.update_parameters()

    def mutate_connections(self, genome):
        # TODO: Should the weights of a disabled gene be mutated? It matters for the speciation
        for connection in (g for g in genome.connections if not g['dis']):
            if random.uniform() < self.mutate_connections_chance:
                if random.uniform() < self.weight_perturb_chance:
                    connection['weight'] += random.uniform(*self.weight_perturb_range)
                else:
                    connection['weight'] = random.uniform(*self.weight_soft_range)

    def add_new_connection(self, genome, to_node, from_node, weight):
        connection = {'to': to_node,
                      'from': from_node,
                      'weight': random.uniform(*self.weight_soft_range),
                      'dis': False}
        if (to_node, from_node) in self.mutations_list:
            connection['innov'] = self.mutations_list[(to_node, from_node)]
        else:
            connection['innov'] = self.connection_innovation
            self.mutations_list[(to_node, from_node)] = self.connection_innovation
            self.connection_innovation += 1
        genome.connections.append(connection)


    def mutate_add_connection(self, genome, unique=False, repeat=3):
        # Input nodes should not be in the in position of a connection
        # TODO: Should output nodes be allowed in the out position of a connection?
        # TODO: Should new connections between already connected nodes be allowed? (unique=True means NO)
        possible_in_nodes = [-i - 1 for i in range(self.input_size, self.input_size + self.output_size)]
        possible_in_nodes.extend(genome.hidden)
        possible_out_nodes = [-i - 1 for i in range(self.input_size)]
        possible_out_nodes.extend(genome.hidden)
        in_out = (random.choice(possible_in_nodes), random.choice(possible_out_nodes))
        if unique:
            times = 0
            is_unique = True
            for gene in genome.connections:
                if gene['to'] == in_out[0] and gene['from'] == in_out[1]:
                    is_unique = False
                    break
            while not is_unique and times < repeat:
                is_unique = True
                in_out = (random.choice(possible_in_nodes), random.choice(possible_out_nodes))
                for gene in genome.connections:
                    if gene['to'] == in_out[0] and gene['from'] == in_out[1]:
                        is_unique = False
                        break
                times += 1
            if not is_unique:
                return False
        self.add_new_connection(genome, in_out[0], in_out[1], random.uniform(*self.weight_soft_range))
        if unique:
            return True

    def mutate_add_node(self, genome):
        available_connections = [c for c in genome.connections if not c['dis']]
        if not available_connections:
            return
        split_connection = random.choice(available_connections)
        split_connection['dis'] = True
        repeats = split_connection['innov'] in self.mutations_list
        if repeats:
            new_node = self.mutations_list[split_connection['innov']]
        else:
            new_node = self.node_innovation
            self.mutations_list[split_connection['innov']] = self.node_innovation
            self.node_innovation += 1
        self.add_new_connection(genome, split_connection['to'], new_node, 1)
        genome.hidden.append(new_node)

    def mutate_add_input(self, genome):
        initial_input_index = random.randint(self.input_size)
        initial_input = initial_input_index - self.input_size
        i = 0
        is_connected = genome.is_connected(initial_input)
        while i < self.input_size and is_connected:
            initial_input_index = (initial_input_index + 1) % self.input_size
            initial_input = initial_input_index - self.input_size
            is_connected = genome.is_connected(initial_input)
            i += 1
        if not is_connected:
            for i in range(-self.output_size - self.input_size, -self.input_size):
                self.add_new_connection(genome, i, initial_input, random.uniform(*self.weight_soft_range))


    def mutate_genome(self, genome):
        self.mutate_connections(genome)
        if random.uniform() < self.new_connection_chance:
            self.mutate_add_connection(genome)
        if random.uniform() < self.new_node_chance:
            self.mutate_add_node(genome)
        if random.uniform() < self.new_input_chance:
            self.mutate_add_input(genome)

    def mutate_population(self):
        self.mutations_list = {}
        for species in self.species:
            elitist = len(species.population) > self.elitist_threshold
            if elitist:
                for genome in species.population[1:]:
                    self.mutate_genome(genome)
            else:
                for genome in species.population:
                    self.mutate_genome(genome)

    def crossover(self, pair):
        # If the pair is a genome with itself the a copy of the genome will be immediately returned
        if pair[0] is pair[1]:
            return pair[0].copy()
        # The new genome will initially inherit the species of the more fit parent
        # TODO: Consider whether to use shared fitness or real fitness
        pair = sorted(pair, key=lambda x: x.fitness, reverse=True)
        baby_genome = NEATGenome.reproduce(*pair)
        # Crossover the connections of the pair
        mother_genome = pair[0]
        i_mother = 0
        len_mother = len(mother_genome.connections)
        father_genome = pair[1]
        i_father = 0
        len_father = len(father_genome.connections)
        equals = father_genome.fitness == mother_genome.fitness
        baby_hidden_set = set()
        while i_mother < len_mother or i_father < len_father:
            if i_mother < len_mother and i_father < len_father:
                if mother_genome.connections[i_mother]['innov'] == father_genome.connections[i_father]['innov']:
                    # If the same innovation exists in both parents, choose randomly
                    # TODO: For now disabled status is inherited from the mother unless both parents are equal
                    if random.randint(2):
                        baby_genome.connections.append(dict(mother_genome.connections[i_mother]))
                        if father_genome.connections[i_father]['dis'] and equals:
                            baby_genome.connections[-1]['dis'] = True
                    else:
                        baby_genome.connections.append(dict(father_genome.connections[i_father]))
                        if mother_genome.connections[i_mother]['dis']:
                            baby_genome.connections[-1]['dis'] = True
                        elif not equals:
                            baby_genome.connections[-1]['dis'] = False
                    i_mother += 1
                    i_father += 1
                elif mother_genome.connections[i_mother]['innov'] < father_genome.connections[i_father]['innov']:
                    # If the connection of the mother is disjoint it will be inherited as it is the fitter parent.
                    baby_genome.connections.append(dict(mother_genome.connections[i_mother]))
                    i_mother += 1
                else:
                    # In the case that the fitness of the father equals that of the mother disjoint connections of
                    # the father will also be inherited.
                    if father_genome.fitness == mother_genome.fitness:
                        baby_genome.connections.append(dict(father_genome.connections[i_father]))
                        i_father += 1
                    else:
                        # In this case no new connection was added so there is no point to add the last one to the set
                        # It can also cause the code to crush if this is the first connection
                        # i.e. baby_genome.connections == []
                        i_father += 1
                        continue
            elif i_mother < len_mother:
                # Excess connections of the mother are inherited
                baby_genome.connections.append(dict(mother_genome.connections[i_mother]))
                i_mother += 1
            elif equals and i_father < len_father:
                # Excess connections of the father are inherited only if its fitness equals that of the mother
                baby_genome.connections.append(dict(father_genome.connections[i_father]))
                i_father += 1
            else:
                # If the mother genome is finished and the father is not as fit as the mother then we finished
                break
            if baby_genome.connections[-1]['to'] >= 0:
                baby_hidden_set.add(baby_genome.connections[-1]['to'])
            if baby_genome.connections[-1]['from'] >= 0:
                baby_hidden_set.add(baby_genome.connections[-1]['from'])
        baby_genome.hidden = sorted(baby_hidden_set)
        return baby_genome

    def crossover_population(self):
        current_population_size = len([g for s in self.species for g in s.population])
        required_babies = self.population_size - current_population_size
        breedable_species = [s for s in self.species if s.unchanged_time < self.stagnation_time]
        # In the case where all the species are stagnant, we will continue with the two best ones only
        if not breedable_species:
            breedable_species = self.species[:2]
        total_species_fitness = sum((s.fitness for s in breedable_species))
        total_babies_so_far = 0
        for species in breedable_species[1:]:
            num_babies = int(required_babies * species.fitness / total_species_fitness)
            total_babies_so_far += num_babies
            for i in range(num_babies):
                # Randomly choose if to do an inter-species crossover or regular one
                if random.uniform() < self.interspecies_crossover_chance:
                    # Choose two unique species from the breedable ones, at least one will differ from the current one
                    option_species = sample(breedable_species, 2)
                    other_species = option_species[0] if option_species[0] is not species else option_species[1]
                    pair = (random.choice(species.population), random.choice(other_species.population))
                else:
                    # Choose two individuals from the species
                    pair = random.choice(species.population, 2)
                baby_genome = self.crossover(pair)
                # Add the genome to its assigned species (Might be the other species)
                baby_genome.species.add_genome(baby_genome)
        # The last species (Which is the most fit one) will get the rest of the required babies
        species = self.species[0]
        num_babies = required_babies - total_babies_so_far
        for i in range(num_babies):
            if len(breedable_species) > 1 and random.uniform() < self.interspecies_crossover_chance:
                option_species = sample(breedable_species, 2)
                other_species = option_species[0] if option_species[0] is not species else option_species[1]
                pair = (random.choice(species.population), random.choice(other_species.population))
            else:
                pair = random.choice(species.population, 2)
            baby_genome = self.crossover(pair)
            baby_genome.species.add_genome(baby_genome)

    def compatibility_distance(self, genome, other_genome):
        disjoint_count = 0
        excess_count = 0
        weight_difference = 0
        matching_count = 0
        i_genome = 0
        len_genome = len(genome.connections)
        i_other = 0
        len_other = len(other_genome.connections)
        normalization_factor = max(len_genome, len_other)
        # if normalization_factor < 20:
        #     normalization_factor = 1
        while i_genome < len_genome or i_other < len_other:
            if i_genome < len_genome and i_other < len_other:
                if genome.connections[i_genome]['innov'] == other_genome.connections[i_other]['innov']:
                    weight_difference += np.abs(genome.connections[i_genome]['weight'] -
                                                other_genome.connections[i_other]['weight'])
                    matching_count += 1
                    i_genome += 1
                    i_other += 1
                elif genome.connections[i_genome]['innov'] < other_genome.connections[i_other]['innov']:
                    disjoint_count += 1
                    i_genome += 1
                else:
                    disjoint_count += 1
                    i_other += 1
            elif i_genome < len_genome:
                excess_count += len_genome - i_genome
                break
            else:
                excess_count += len_other - i_other
                break
        if matching_count > 0:
            distance = (self.disjoint_coeff * disjoint_count + self.excess_coeff * excess_count) / normalization_factor
            distance += self.weights_coeff * weight_difference / matching_count
        else:
            distance = np.inf
        return distance

    def speciate_genome(self, genome):
        # First the genome will be tested against it's already assigned species
        distance = self.compatibility_distance(genome, genome.species.representative)
        if distance < self.compatibility_threshold:
            return
        # Then the genome will be tested against the rest of the species
        genome.species.population.remove(genome)
        for species in self.species:
            distance = self.compatibility_distance(genome, species.representative)
            if distance < self.compatibility_threshold:
                genome.species = species
                species.add_genome(genome)
                return
        # Then a new species will be created for the genome and added to the environment
        new_species = NEATSpecies(self)
        new_species.add_genome(genome)
        genome.species = new_species
        self.species.append(new_species)

    def speciate_population(self):
        for genome in (g for s in self.species for g in s.population[1:]):
            self.speciate_genome(genome)

    def filter_population(self):
        population = sorted((g for s in self.species for g in s.population), key=lambda g: g.fitness, reverse=True)
        for genome in population[int(len(population) * self.filter_percentage):]:
            genome.species.population.remove(genome)
            if len(genome.species.population) == 0:
                self.species.remove(genome.species)

    def fit_function(self, genome):
        graph = NEATGraphModel(NEATGraph.from_genome(genome))
        player = graph.get_agent()
        return FourInARowFitFunction.instance.fit_function(player)

    def fit_population(self):
        for species in self.species:
            total_fit = 0
            species_size = len(species.population)
            for genome in species.population:
                genome.real_fitness = self.fit_function(genome)
                total_fit += genome.real_fitness / species_size
                genome.fitness = self.bolster_percentage * genome.real_fitness / species_size + \
                                 (1 - self.bolster_percentage) * genome.real_fitness
            species.fitness = total_fit
            species.population.sort(key=lambda x: x.fitness, reverse=True)
        self.species.sort(key=lambda x: x.fitness, reverse=True)

    def add_history(self, tag):
        self.history.append((self.generation, tag))

    def record_history(self):
        self.species_history.append([s.id for s in self.species])
        # self.best_genomes.append({s.id: s.best_genome().genesis().to_string() for s in self.species})
        self.species_fit.append({s.id: {'fit': s.fitness,
                                        'best': s.population[0].real_fitness,
                                        'size': len(s.population)} for s in self.species})
        all_fits = [g.real_fitness for s in self.species for g in s.population]
        self.fit_history['max'].append(max(all_fits))
        self.fit_history['avg'].append(np.mean(all_fits))
        self.fit_history['min'].append(min(all_fits))

    def update_parameters(self):
        for species in self.species:
            species.update_parameters()

        # Start adjusting compatibility threshold only after some generations
        if self.threshold_adjustment_param and self.generation > 5:
            if len(self.species) > self.target_species_number:
                self.compatibility_threshold += self.threshold_adjustment_param
            elif len(self.species) < self.target_species_number:
                self.compatibility_threshold -= self.threshold_adjustment_param
                if self.compatibility_threshold < self.threshold_adjustment_param:
                    self.compatibility_threshold = self.threshold_adjustment_param

        if self.bolster_percentage > self.minimum_bolster:
            self.bolster_percentage *= self.bolster_reduce_rate
            if self.bolster_percentage < self.minimum_bolster:
                self.bolster_percentage = self.minimum_bolster

    def best_species(self):
        return self.species[0]

    def worst_species(self):
        return self.species[-1]

    def best_genome(self):
        return max((g for s in self.species for g in s.population), key=lambda x: x.real_fitness)

    def breed(self):
        self.generation += 1
        self.filter_population()
        self.crossover_population()
        self.mutate_population()
        self.speciate_population()
        self.fit_population()
        self.record_history()
        self.update_parameters()

    def save(self, file):
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        return pickle.load(file)


import matplotlib.pyplot as plt
from MonteCarloTree import test_agent


def lerp_color_rg(num, bottom, top):
    if num < bottom:
        num = bottom
    if num > top:
        num = top
    perc = (num - bottom) / (top - bottom)
    red = int((1 - perc) * 255)
    green = int(perc * 255)
    color = '#' + hex(red)[2:] + hex(green)[2:] + '00'
    return color


def graph_genome(genome, graph_disabled=True):
    from graphviz import Digraph

    g = Digraph('G', filename='hello.gv')
    g.attr(rankdir='LR', size='8,8')
    # print(g)

    # Cluster the inputs and the outputs
    with g.subgraph(name='cluster_sensors') as c:
        for n in range(-genome.sensors, 0):
            c.node(str(n))

    with g.subgraph(name='cluster_outputs') as c:
        for n in range(-genome.sensors - genome.outputs, -genome.sensors):
            c.node(str(n))

    # Add all the hidden nodes
    with g.subgraph(name='cluster_hidden') as c:
        for n in genome.hidden:
            c.node(str(n))

    # Add all the connections:
    for c in genome.connections:
        if c['dis']:
            if graph_disabled:
                g.attr('edge', color='lightgrey')
            else:
                continue
        else:
            color = lerp_color_rg(c['weight'], -3, 3)
            g.attr('edge', color=color)
        g.edge(str(c['from']), str(c['to']))  # , label=str(c['weight']))

    g.view()


def analyze_environment(env):
    plt.grid(True)
    plt.plot(env.fit_history['max'])
    plt.plot(env.fit_history['avg'])
    plt.plot(env.fit_history['min'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

    # plt.close()
    plt.grid(True)
    plt.plot([len(s) for s in env.species_history])
    plt.xlabel('Generations')
    plt.ylabel('Species')
    plt.show()

    plt.grid(True)
    y = np.zeros((env.species_id, env.generation + 1))
    for i, all_s in enumerate(env.species_fit):
        for s in env.species_history[i]:
            y[s, i] = all_s[s]['size']
    plt.stackplot(list(range(env.generation + 1)), y)
    plt.xlabel('Generations')
    plt.ylabel('Population')
    plt.show()

if __name__ == '__main__':
    with open('neat_e_2.pkl', 'rb') as file:
        neat_e = NEATEnvironment.load(file)

    # neat_e = NEATEnvironment(150, 7 * 6 + 1, 7, 0.1, 0.05, 0.5, 15, 1, 0.6, 0.005, 0.001, 5, 0.5, 0.05, 0.03, 0.001,
    #                          0.9, [-2, 2], [-0.1, 0.1], 4, 1, 1, 3, 5, 0)

    while True:
        # pa = NEATGraphModel(neat_e.best_genome().genesis()).get_agent()

        # print('Test vs opp.horizontal: {0}'.format(test_agent(pa, HorizontalPlayer('horizontal', 7, 6))))
        # print('Test vs opp.const: {0}'.format(test_agent(pa, ConstPlayer('const', 7, 6))))
        # print('Test vs opp.rand: {0}'.format(test_agent(pa, RandomPlayer('rand', 7, 6, True))))
        # print('Test vs opp.min: {0}'.format(test_agent(pa, MinPlayer('min', 7, 6, True))))

        for i in range(2):
            with open('neat_e_2.pkl', 'wb') as file:
                neat_e.save(file)

            # analyze_environment(neat_e)
            # graph_genome(neat_e.best_genome())
            for j in range(10):
                neat_e.breed()
                print(neat_e.generation)

