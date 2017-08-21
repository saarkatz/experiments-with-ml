import numpy as np
from scipy.special import expit

from Optimizers.NNFunctions import back_prop


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
    def __init__(self, activation, size, input_layer=None, has_bias=True):
        self.activation = activation
        self.size = size
        self.is_input = input_layer is None
        self.has_bias = has_bias

        # Internal matrix of weights
        if input_layer:
            self.matrix = init_weight(input_layer.size, size, input_layer.has_bias)
            self.prev_layer = input_layer
        else:
            self.matrix = None
            self.prev_layer = None

    def num_layers(self):
        if self.is_input:
            return 1
        return 1 + self.prev_layer.num_layers()

    def layer_matrices(self):
        if self.is_input:
            return []

        prev_matrices = self.prev_layer.layer_matrices()
        prev_matrices.append(self.matrix)
        return prev_matrices

    def run(self, input_vec):
        is_list_input = isinstance(input_vec, list)

        if self.is_input:
            # Return the values given in input_vec
            return input_vec

        # Get the result of the previous layer
        input_vector = self.prev_layer.run(input_vec)

        # Calculate output vector
        if self.prev_layer.has_bias:
            if is_list_input:
                output_vector = [np.dot(self.matrix, np.concatenate((np.ones(1), input_v)))
                                 for input_v in input_vector]
            else:
                output_vector = np.dot(self.matrix, np.concatenate((np.ones(1), input_vector)))
        else:
            if is_list_input:
                output_vector = [np.dot(self.matrix, input_v) for input_v in input_vector]
            else:
                output_vector = np.dot(self.matrix, input_vector)

        # Use activation function in place
        if is_list_input:
            [self.activation.func_in_place(output_v) for output_v in output_vector]
        else:
            self.activation.func_in_place(output_vector)

        # Return the out vector
        output_vector[np.isnan(output_vector)] = 0
        return output_vector

    def run_all_partial(self, input_vec, activation_after=False):
        is_list_input = isinstance(input_vec, list)

        if self.is_input:
            if is_list_input:
                # Return list of the histories
                return [[input_v] for input_v in input_vec]
            else:
                # Return the input_vec
                return [input_vec]

        # Get the result of the previous layer
        history = self.prev_layer.run_all_partial(input_vec, activation_after)

        if activation_after and not self.prev_layer.is_input:
            if is_list_input:
                a_history = [self.prev_layer.activation.func(h[-1]) for h in history]
            else:
                a_history = self.prev_layer.activation.func(history[-1])
        else:
            a_history = history[-1]

        # Calculate output vector
        if self.prev_layer.has_bias:
            if is_list_input:
                output_vector = [np.dot(self.matrix, np.concatenate((np.ones(1), h)))
                                 for h in a_history]
            else:
                output_vector = np.dot(self.matrix, np.concatenate((np.ones(1), a_history)))
        else:
            if is_list_input:
                output_vector = [np.dot(self.matrix, h) for h in a_history]
            else:
                output_vector = np.dot(self.matrix, a_history)

        if not activation_after:
            # Use activation function in place
            if is_list_input:
                [self.activation.func_in_place(output_v) for output_v in output_vector]
            else:
                self.activation.func_in_place(output_vector)

        # Calculate and return the out vector along with the rest of the history
        if is_list_input:
            [h.append(output_v) for h, output_v in zip(history, output_vector)]
        else:
            history.append(output_vector)
        return history

    def get_weights_as_vector(self):
        if self.is_input:
            return np.zeros(0)

        return np.concatenate((self.prev_layer.get_weights_as_vector(), self.matrix.flatten()))

    def set_weights_from_vector(self, weights_vector):
        if self.is_input:
            return

        size = self.matrix.size
        self.matrix[:, :] = weights_vector[-size:].reshape(self.matrix.shape)
        self.prev_layer.set_weights_from_vector(weights_vector[:-size])

    def learn(self, data_set, learn_rate=1e-3, reward=1, lambda_reg=0.5):
        if reward:
            grad_matrices = back_prop(self, data_set, lambda_reg)
            for layer, grad in zip(self.layer_matrices(), grad_matrices):
                layer -= learn_rate * reward * grad

    def unlearn(self, data_set, learn_rate=1e-3, reward=1, lambda_reg=0.5):
        grad_matrices = back_prop(self, data_set, lambda_reg)
        for layer, grad in zip(self.layer_matrices(), grad_matrices):
            layer += learn_rate * reward * grad

    def _policy_backward(self, input_vec, output_vec, lambda_reg, all_partial=None):
        if self.is_input:
            return []

        if not all_partial:
            all_partial = self.run_all_partial(input_vec, activation_after=True)
            delta_curr = self.activation.func(all_partial[-1]) - output_vec
        else:
            delta_curr = output_vec
        v = np.dot(self.matrix.transpose(), delta_curr)
        if self.prev_layer.has_bias:
            v = v[1:]
        delta_next = v * self.activation.diff(all_partial[-2])
        delta_list = self.prev_layer._policy_backward(input_vec, delta_next, lambda_reg, all_partial[:-1])
        a_prev = self.activation.func(all_partial[-2])
        if self.prev_layer.has_bias:
            a_prev = np.concatenate((np.ones(1), a_prev))
        delta_list.append(np.outer(delta_curr, a_prev))
        # Add regularization term
        if lambda_reg:
            delta_list[-1] += lambda_reg * self.matrix

        return delta_list

    def policy_backward(self, input_vec, output_vec, lambda_reg):
        return self._policy_backward(input_vec, output_vec, lambda_reg)

    def save(self, path):
        np.save(path, self.get_weights_as_vector())

    def load(self, path):
        self.set_weights_from_vector(np.load(path))


def create_layer(activation, size=1, input_layer=None, has_bias=True):
    return NeuralNetwork(activation, size, input_layer, has_bias)


class Sigmoid:
    @staticmethod
    def func_in_place(vec):
        vec[:] = expit(vec)
    @staticmethod
    def func(vec):
        r = expit(vec.copy())
        return r
    @staticmethod
    def diff(vec):
        r = Sigmoid.func(vec)
        return r * (1 - r)


class ReLU:
    @staticmethod
    def func_in_place(vec):
        vec[vec < 0] = 0
    @staticmethod
    def func(vec):
        r = vec.copy()
        r[r < 0] = 0
        return r
    @staticmethod
    def diff(vec):
        r = np.zeros_like(vec)
        r[vec > 0] = 1
        return r


class Identity:
    @staticmethod
    def func_in_place(vec):
        pass
    @staticmethod
    def func(vec):
        return vec.copy()
    @staticmethod
    def diff(vec):
        return np.ones_like(vec)


if __name__ == '__main__':
    np.seterr(all='raise')
    np.random.seed(40)
    input_vector = np.array([[0, 0, 0, 0, 0, 0],
                             # [0, 0, 0, 0, 0, 0],
                             # [0, 0, 0, 0, 0, 0],
                             # [0, 0, 0, 0, 0, 0],
                             # [0, 0, 0, 0, 0, 0],
                             # [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]]).flatten()
    output_vector = np.array([0, 0, 0, 1, 0, 0, 0])

    data_set = [(input_vector, output_vector)]

    # print(out.run(input_vector))
    # print(out.get_weights_as_vector())

    # print(wrapped_cost_function(out.get_weights_as_vector(), out, data_set, 0.5))

    vec = np.zeros(11)
    for j in range(10):
        x = create_layer(None, 6 * 2, has_bias=True)
        w1 = create_layer(Sigmoid, 3 * 7, x, has_bias=False)
        w2 = create_layer(Sigmoid, 2 * 7, w1, has_bias=False)
        out = create_layer(Identity, 7, w2)

        for i in range(0, 10001):
            out.learn(data_set, reward=1, lambda_reg=0.1)
            if i % 1000 == 0:
                result = out.run(input_vector)
                # print(result)
                vec[int(i/1000)] += np.linalg.norm(output_vector - result)
    print(vec/10)

    print(out.run(input_vector))

    #out.set_weights_from_vector(np.array([0,1,1,1,-1,1,1,1,-1,0,0,0,0,-1,0,0,0]))

    # print(out.run(input_vector})
    # print(out.get_weights_as_vector())

    # check gradients
    # check_gradients(compute_numerical_gradient(out.get_weights_as_vector(), out, [(input_vector, output_vector)]), wrapped_back_prop(out.get_weights_as_vector(), out, [(input_vector, output_vector)]))

