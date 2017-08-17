import numpy as np
#Function inputs are:
#l_in: size of the input layer for the current layer.
#l_out: size of the output layer for the current layer.
#returns random weights matrix
def init_weight (l_in, l_out):
    epsilon_init = np.sqrt(6) / (np.sqrt(l_in + l_out))
    weights = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return weights

def back_prop (, ):
    return

# Function gets set of training examples and returns the cost function
def cost_function(nn, data_set):
    j = 0
    for input_vec, output in data_set:
        run_res = nn.run(input_vec)
        for k in range(input_vec.size):
            j += -output[k]*np.log(run_res[k]) + (1-output[k])*np.log(run_res[k])
    j *= (1/len(data_set))
    return j
