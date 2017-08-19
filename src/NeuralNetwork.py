import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_cg
from NNFunctions import wrapped_cost_function, back_prop, compute_numerical_gradient, check_gradients


# Function inputs are:
# l_in: size of the input layer for the current layer.
# l_out: size of the output layer for the current layer.
# returns random weights matrix
def init_weight(l_in, l_out, add_bias):
    if add_bias:
        l_in += 1
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, l_in) * 2 * epsilon_init - epsilon_init
    return weights


class NeuralNetwork:
    def __init__(self, name, size, input_layer=None, is_placeholder=False, has_bias=True):
        self.name = name
        self.size = size
        self.has_bias = has_bias

        # Place holder are meant as layers that can receive input
        self.is_placeholder = is_placeholder
        self.placeholders = input_layer.placeholders if input_layer else []
        if is_placeholder:
            self.placeholders.append(name)

        # Internal matrix of weights
        if input_layer:
            self.matrix = init_weight(input_layer.size, size, input_layer.has_bias)
            self.prev_layer = input_layer
        else:
            self.matrix = None
            self.prev_layer = None

    def num_layers(self):
        if self.is_placeholder:
            return 1

        return 1 + self.prev_layer.num_layers()

    def layer_matrices(self):
        if self.is_placeholder:
            return []

        prev_matrices = self.prev_layer.layer_matrices()
        prev_matrices.append(self.matrix)
        return prev_matrices

    def run(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            return input_dict[self.name]

        # Get the result of the previous layer
        input_vector = self.prev_layer.run(input_dict)

        # Calculate output vector
        if self.prev_layer.has_bias:
            output_vector = np.dot(self.matrix, np.concatenate((np.ones(1), input_vector)))
        else:
            output_vector = np.dot(self.matrix, input_vector)

        # Return the out vector
        return expit(output_vector)

    def run_all_partial(self, input_dict):
        if self.is_placeholder:
            # Return the values given in input_dict
            if self.has_bias:
                return [np.concatenate((np.ones(1), input_dict[self.name]))]
            else:
                return [input_dict[self.name]]

        # Get the result of the previous layer
        history = self.prev_layer.run_all_partial(input_dict)

        # Calculate output vector
        output_vector = expit(np.dot(self.matrix, history[0]))
        if self.has_bias:
            full_output_vector = np.concatenate((np.ones(1), output_vector))
        else:
            full_output_vector = output_vector

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

    def learn(self, data_set, learn_rate=1e-3, reward=1.0, lambda_reg=0.5):
        grad_matrices = back_prop(self, data_set, lambda_reg)
        for layer, grad in zip(self.layer_matrices(), grad_matrices):
            layer -= learn_rate * reward * grad

    def unlearn(self, data_set, learn_rate=1e-3, reward=1.0, lambda_reg=0.5):
        grad_matrices = back_prop(self, data_set, lambda_reg)
        for layer, grad in zip(self.layer_matrices(), grad_matrices):
            layer += learn_rate * reward * grad

    def save(self, path):
        np.save(path, self.get_weights_as_vector())

    def load(self, path):
        self.set_weights_from_vector(np.load(path))


def create_placeholder(name, size=1, has_bias=True):
    return NeuralNetwork(name, size, None, True, has_bias=has_bias)


def create_dense_layer(name, size, input_layer, has_bias=True):
    return NeuralNetwork(name, size, input_layer, False, has_bias=True)


if __name__ == '__main__':
    input_vector = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]]).flatten()
    output_vector = np.array([0, 0, 0, 1, 0, 0, 0])

    data_set = [(input_vector, output_vector)]

    # print(out.run({'input': input_vector}))
    # print(out.get_weights_as_vector())

    # print(wrapped_cost_function(out.get_weights_as_vector(), out, data_set, 0.5))

    vec = np.zeros(11)
    for j in range(10):
        x = create_placeholder('input', 6 * 7)
        w1 = create_dense_layer('W1', 3 * 7, x)
        w2 = create_dense_layer('W2', 2 * 7, w1)
        out = create_dense_layer('out', 7, w2)

        for i in range(0, 10001):
            out.learn(data_set, reward=-1e-1, lambda_reg=0.1)
            if i % 1000 == 0:
                result = out.run({'input': input_vector})
                # print(result)
                vec[int(i/1000)] += np.linalg.norm(1 - output_vector - result)
    print(vec/10)

    print(out.run({'input': input_vector}))

    #out.set_weights_from_vector(np.array([0,1,1,1,-1,1,1,1,-1,0,0,0,0,-1,0,0,0]))

    # print(out.run({'input': input_vector}))
    # print(out.get_weights_as_vector())

    # check gradients
    # check_gradients(compute_numerical_gradient(out.get_weights_as_vector(), out, [(input_vector, output_vector)]), wrapped_back_prop(out.get_weights_as_vector(), out, [(input_vector, output_vector)]))

