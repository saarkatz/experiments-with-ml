import numpy as np


# Function inputs are:
# l_in: size of the input layer for the current layer.
# l_out: size of the output layer for the current layer.
# returns random weights matrix
def init_weight (l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights


# Function gets set of training examples and returns the cost function
def cost_function(nn, data_set, lambda_reg = 0.5):
    j = 0
    m = len(data_set)
    for input_vec, output in data_set:
        run_res = nn.run({'input': input_vec})
        for k in range(input_vec.size):
            j += output[k]*np.log(run_res[k]) + (1-output[k])*np.log(1-run_res[k])
    j *= (-1/m)
    current_layer = nn
    regularization = 0
    while not current_layer.is_placeholder:
        theta_vec = current_layer.matrix.flatten()
        for k in (1, theta_vec.size-1):
            regularization += np.power(theta_vec[k], 2)
        current_layer = current_layer.prev_layer
    regularization *= lambda_reg / (2 * m)
    j += regularization
    return j


def wrapped_cost_function(weights, nn, dataset):
    nn.set_weights_from_vector(weights)
    return cost_function(nn, dataset)


def sigmoid_direvative_with_bias(vec):
    out_vec = vec * (1 - vec)
    out_vec[0] = 1
    return out_vec


# Suppose to be back prog
def back_prop(nn, input_dict):
    a = nn.run_all_partial(input_dict)
    y = np.zeros(a[0].shape)
    y[np.argmax(a[0][1:]) + 1] = 1
    delta_vec_prev = a[0] - y
    curr_layer = nn
    d = []
    for k in range(1, len(a) - 1):
        delta_vec_next = np.dot(np.transpose(nn.matrix), delta_vec_prev[1:]) * sigmoid_direvative_with_bias(a[k])
        d.append(np.outer(delta_vec_prev, a[k])[1:])
        delta_vec_prev = delta_vec_next
        curr_layer = curr_layer.prev_layer
    else:
        d.append(np.outer(delta_vec_prev, a[-1])[1:])

    return d

def wrapped_back_prop(x, nn, input_vector):
    nn.set_weights_from_vector(x)
    d = back_prop(nn, {'input': input_vector})
    grad = np.zeros(0)
    for layer_d in d:
        grad = np.concatenate((layer_d.flatten(), grad))
    return grad