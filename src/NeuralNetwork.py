import numpy as np
from scipy.special import expit

#Function inputs are:
#l_in: size of the input layer for the current layer.
#l_out: size of the output layer for the current layer.
#returns random weights matrix
def init_weight (l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights

class NeuralNetwork:
    def __init__(self, name, size, input_layer=None, is_placeholder=False):
        self.name = name
        self.size = size

        # Place holder are meant as layers that can receive input
        self.is_placeholder = is_placeholder
        self.placeholders = input_layer.placeholders if input_layer else []
        if is_placeholder:
            self.placeholders.append(name)

        # Internal matrix of weights
        if input_layer:
            self._matrix = init_weight(input_layer.size, size)
            self._prev_layer = input_layer

    def run(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return input_dict[self.name]

        # Get the result of the previous layer
        input_vector = self._prev_layer.run(input_dict)

        # Concatenate bias value
        full_input_vector = np.concatenate((np.ones((1,)), input_vector))

        # Calculate and return the out vector
        return expit(np.dot(self._matrix, full_input_vector))

    def _run_all_partial(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return [input_dict[self.name]], [input_dict[self.name]]

        # Get the result of the previous layer
        history_z, history_a = self._prev_layer._run_all_partial(input_dict)

        # Concatenate bias value
        full_input_vector = np.concatenate((np.ones((1,)), history_a[-1]))

        # Calculate and return the out vector along with the rest of the history
        history_z.insert(0, np.dot(self._matrix, full_input_vector))
        history_a.insert(0, expit(history_z[-1]))
        return history_z, history_a


def create_placeholder(name, size=1):
    return NeuralNetwork(name, size, None, True)


def create_dense_layer(name, size, input_layer):
    return NeuralNetwork(name, size, input_layer, False)


def create_const(name, size, const):
    nn = NeuralNetwork(name, size, None, False)
    nn.run = lambda x: np.ones((size,)) * const
    nn._run_all_partial = lambda x: [np.ones((size,)) * const], [np.ones((size,)) * const]
    return nn

def add_networks(network1, network2):
    nn = NeuralNetwork('+', network1.size, None, False)
    def run(input_dict):
        # Get the result of the previous layer
        input_vector1 = network1.run(input_dict)
        input_vector2 = network2.run(input_dict)

        return input_vector1 + input_vector2

    def _run_all_partial(input_dict):
        history_z_1, history_a_1 = network1._run_all_partial(input_dict)
        history_z_2, history_a_2 = network2._run_all_partial(input_dict)

        out_vector = history_a_1[-1] + history_a_2[-1]
        history_z_1.extend(history_z_2)
        history_a_1.extend(history_a_2)
        history_z_1.append(out_vector)
        history_a_1.append(out_vector)

        return history_z_1, history_a_1

    nn.run = run
    nn._run_all_partial = _run_all_partial
    return nn


if __name__ == '__main__':
    a = create_placeholder('input', 1)
    b = create_dense_layer('b', 1, a)

    print(b._run_all_partial({'input': np.array([2])}))