import numpy as np
#Function inputs are:
#l_in: size of the input layer for the current layer.
#l_out: size of the output layer for the current layer.
#returns random weights matrix
def init_weight (l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights

