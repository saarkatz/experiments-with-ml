import numpy as np

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

        # internal matrix of weights
        self._matrix = np.zeros(shape=(size, input_layer.size), dtype=np.float32)
        self._prev_layer = input_layer


def create_placeholder(name, size=1):
    return NeuralNetwork(name, size, None, True)


def create_dense_layer(name, size, input_layer):
    return NeuralNetwork(name, size, input_layer, False)


