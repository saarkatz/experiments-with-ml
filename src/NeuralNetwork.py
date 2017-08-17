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

        # Calculate and return the out vector
        return np.dot(self._matrix, input_vector)

    def _run_all_partial(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return [input_dict[self.name]]

        # Get the result of the previous layer
        history = self._prev_layer.run(input_dict)

        # Calculate and return the out vector along with the rest of the history
        history.append(np.dot(self._matrix, history[-1]))
        return history


def create_placeholder(name, size=1):
    return NeuralNetwork(name, size, None, True)


def create_dense_layer(name, size, input_layer):
    return NeuralNetwork(name, size, input_layer, False)

def create_const(name, size):
    
