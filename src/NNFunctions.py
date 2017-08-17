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

# Suppose to be back prog
def back_prop (self, input_dic):
    a, z = _run_all_partial(input_dic)
    lambda_vec = [None] * (len(a)-1)
    delta_vec = [None] * (len(a)-1)
    grad_vec = [None] * (len(a)-1)
    y = np.zeros(len(a[0]))
    y[np.argmax(a[0])] = 1
    lambda_vec[0] = np.array(a[0]) - y
    current_layer = self._prev_layer
    for k in range(lambda_vec[1:].size):
        lambda_vec[k+1] = np.multiply((lambda_vec[k]*current_layer._matrix), np.multiply(expit(z[k+1]), 1-expit(z[k+1])))
        lambda_vec[k+1] = lambda_vec[k+1][1:]
        delta_vec[k] = lambda_vec[k+1]*np.matrix.transpose(a[k])
        grad_vec[k] = (1 / len(input_dic)) * delta_vec[k]
    return grad_vec

# Function gets set of training examples and returns the cost function
def cost_function(nn, data_set):
    j = 0
    for input_vec, output in data_set:
        run_res = nn.run(input_vec)
        for k in range(input_vec.size):
            j += -output[k]*np.log(run_res[k]) + (1-output[k])*np.log(run_res[k])
    j *= (1/len(data_set))
    return j
