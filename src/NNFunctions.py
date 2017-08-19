import numpy as np
import matplotlib.pyplot as pp

# Function inputs are:
# l_in: size of the input layer for the current layer.
# l_out: size of the output layer for the current layer.
# returns random weights matrix
def init_weight(l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights


# Function gets set of training examples and returns the cost function
def cost_function(nn, data_set, lambda_reg=0.5):
    j = 0
    m = len(data_set)
    for input_vec, output in data_set:
        run_res = nn.run({'input': input_vec})
        for k in range(run_res.size):
            j -= output[k]*np.log(run_res[k]) + (1-output[k])*np.log(1-run_res[k])
    j *= 1/m
    current_layer = nn
    regularization = 0
    while not current_layer.is_placeholder:
        weights_vec = np.copy(current_layer.matrix)
        weights_vec = weights_vec[:, 1:]
        weights_vec *= weights_vec
        regularization += sum(sum(weights_vec))
        current_layer = current_layer.prev_layer
    regularization *= lambda_reg / (2 * m)
    j += regularization
    return j


def wrapped_cost_function(weights, nn, dataset, lambda_reg=0.5):
    nn.set_weights_from_vector(weights)
    return cost_function(nn, dataset, lambda_reg)


def sigmoid_direvative_with_bias(vec):
    out_vec = vec * (1 - vec)
    out_vec[0] = 1
    return out_vec


# Suppose to be back prop
def back_prop(nn, data_set, lambda_reg=0.5):
    layer_matrices = nn.layer_matrices()
    delta_matrices = [np.zeros(m.shape) for m in layer_matrices]
    num_layers = nn.num_layers()
    for input_vector, output_vector in data_set:
        a = nn.run_all_partial({'input': input_vector})
        y = np.concatenate((np.ones(1), output_vector))
        delta_vec_prev = a[0] - y
        curr_layer = nn
        for k in range(1, num_layers - 1):
            curr_delta = np.outer(delta_vec_prev, a[k])[1:]
            delta_matrices[-k] += curr_delta

            delta_vec_next = \
                np.dot(np.transpose(curr_layer.matrix), delta_vec_prev[1:]) * sigmoid_direvative_with_bias(a[k])
            delta_vec_prev = delta_vec_next
            curr_layer = curr_layer.prev_layer
        else:
            curr_delta = np.outer(delta_vec_prev, a[-1])[1:]
            delta_matrices[0] += curr_delta

        if lambda_reg:
            # Add regularization term
            for layer_matrix, delta_matrix in zip(layer_matrices, delta_matrices):
                delta_matrix /= len(data_set)
                reg_matrix = np.copy(layer_matrix)
                reg_matrix[:, :1] = 0
                delta_matrix += lambda_reg * reg_matrix

    return delta_matrices


def wrapped_back_prop(x, nn, data_set, lambda_reg=0.5):
    nn.set_weights_from_vector(x)
    d = back_prop(nn, data_set, lambda_reg)
    grad = np.zeros(0)
    for layer_d in d:
        grad = np.concatenate((grad, layer_d.flatten()))
    return grad


def compute_numerical_gradient(x, nn, data_set):
    e = 1e-4
    peturb = np.zeros(x.size)
    grad = np.zeros(x.size)
    for k in range(x.size):
        peturb[k] = e
        loss1 = wrapped_cost_function(x - peturb, nn, data_set)
        loss2 = wrapped_cost_function(x + peturb, nn, data_set)
        grad[k] = (loss2 - loss1) / (2*e)
        peturb[k] = 0
    return grad


def check_gradients(numeric_grad, back_prop_grad):
    diff = np.linalg.norm(numeric_grad-back_prop_grad) / np.linalg.norm(numeric_grad+back_prop_grad)
    print(diff)
    return


def learning_curve(input_vec, output_vec, weights, nn):
    error_train = wrapped_cost_function(weights, nn, [(input_vec, output_vec)])
    return error_train


def calc_cost_iter(weights, nn, dataset, iter_num, cost_vector):
    current_cost = wrapped_cost_function(weights, nn, dataset)
    cost_vector[iter_num] = current_cost


def plot_cost_iter_graph(cost_vec):
    pp.xlabel('Iterations')
    pp.ylabel('Cost Function')
    ax=pp.subplot(111)
    ax.set_xlim(1, cost_vec.size)
    cost = np.arange(1, cost_vec.size, 1)
    ax.plot(cost_vec, color='r', linewidth=1, label="Test")
    pp.xticks(cost)
    pp.grid()
    pp.show()


if __name__ == '__main__':
    a = np.array([5, 3, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05])
    plot_cost_iter_graph(a)