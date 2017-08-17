import numpy as np
#Function inputs are:
#l_in: size of the input layer for the current layer.
#l_out: size of the output layer for the current layer.
#returns random weights matrix
def init_weight (l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights

def back_prop (output, exp_out):
    return

def cost_function(self, data_set):
    j = 0
    for input, output in data_set:
        for k in input.size:
            j += -output[k]*np.log(run(input)[k]) + (1-output[k])*np.log(run(input)[k])
    j *= (1/len(data_set))
    return j
