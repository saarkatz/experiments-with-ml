import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_cg
from NNFunctions import wrapped_cost_function, wrapped_back_prop


# Function inputs are:
# l_in: size of the input layer for the current layer.
# l_out: size of the output layer for the current layer.
# returns random weights matrix
def init_weight(l_in, l_out):
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
            self.matrix = init_weight(input_layer.size, size)
            self.prev_layer = input_layer
        else:
            self.matrix = None
            self.prev_layer = None

    def run(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return input_dict[self.name]

        # Get the result of the previous layer
        input_vector = self.prev_layer.run(input_dict)

        # Concatenate bias value
        full_input_vector = np.concatenate((np.ones((1,)), input_vector))

        # Calculate output vector
        output_vector = np.dot(self.matrix, full_input_vector)

        # Return the out vector
        return expit(output_vector)

    def run_all_partial(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return [np.concatenate((np.ones(1), input_dict[self.name]))]

        # Get the result of the previous layer
        history = self.prev_layer.run_all_partial(input_dict)

        # Calculate output vector
        output_vector = expit(np.dot(self.matrix, history[0]))
        full_output_vector = np.concatenate((np.ones(1), output_vector))

        # Calculate and return the out vector along with the rest of the history
        history.insert(0, full_output_vector)
        return history

    def get_weights_as_vector(self):
        if self.is_placeholder:
            return np.zeros(0)

        return np.concatenate((self.prev_layer.get_weights_as_vector(), self.matrix.flatten()))

    def set_weights_from_vector(self, weights_vector):
        if self.is_placeholder:
            return

        size = self.matrix.size
        self.matrix[:, :] = weights_vector[-size:].reshape(self.matrix.shape)
        self.prev_layer.set_weights_from_vector(weights_vector[:-size])

    def learn(self, input_vector, output_vector, iterations=None):
        xopt = fmin_cg((lambda x: wrapped_cost_function(x, self, [(input_vector, output_vector)])),
                       self.get_weights_as_vector(),
                       lambda x: wrapped_back_prop(x, self, input_vector),
                       maxiter=iterations,
                       callback=lambda x: print(wrapped_cost_function(x, self, [(input_vector, output_vector)])))
        self.set_weights_from_vector(xopt)

    def save(self, path):
        np.save(path, self.get_weights_as_vector())

    def load(self, path):
        self.set_weights_from_vector(np.load(path))


def create_placeholder(name, size=1):
    return NeuralNetwork(name, size, None, True)


def create_dense_layer(name, size, input_layer):
    return NeuralNetwork(name, size, input_layer, False)


# def create_const(name, size, const):
#     nn = NeuralNetwork(name, size, None, False)
#     nn.run = lambda x: np.ones((size,)) * const
#     nn.run_all_partial = lambda x: [np.concatenate((np.ones(1), np.ones((size,)) * const))]
#     return nn
#
#
# def add_networks(network1, network2):
#     nn = NeuralNetwork('+', network1.size, None, False)
#
#     def run(input_dict):
#         # Get the result of the previous layer
#         input_vector1 = network1.run(input_dict)
#         input_vector2 = network2.run(input_dict)
#
#         return input_vector1 + input_vector2
#
#     def _run_all_partial(input_dict):
#         history_1 = network1.run_all_partial(input_dict)
#         history_2 = network2.run_all_partial(input_dict)
#
#         out_vector = history_1[0][1:] + history_2[0][1:]
#         full_output_vector = np.concatenate((np.ones(1), out_vector))
#         history_1.extend(history_2)
#         history_1.append(full_output_vector)
#
#         return history_1
#
#     nn.run = run
#     nn.run_all_partial = _run_all_partial
#     return nn


if __name__ == '__main__':
    # a = create_const('a', 3, 3)
    # b = create_const('b', 3, 2)
    # c = create_dense_layer('c', 3, b)

    x = create_placeholder('input', 6 * 7)
    W1 = create_dense_layer('W1', 50, x)
    W2 = create_dense_layer('W2', 50, W1)
    out = create_dense_layer('out', 7, W2)

    input_vector = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]]).flatten()
    output_vector = np.array([0, 0, 0, 1, 0, 0, 0])

    print(out.run({'input': input_vector}))
    # print(out.get_weights_as_vector())

    out.learn(input_vector, output_vector, iterations=100)

    # out.set_weights_from_vector(np.array([0,1,1,1,-1,1,1,1,-1,0,0,0,0,-1,0,0,0]))

    print(out.run({'input': input_vector}))
    # print(out.get_weights_as_vector())
