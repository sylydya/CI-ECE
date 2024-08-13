import os
import numpy as np

def get_settings(setting):
    if setting == "simu1":
        beta_list = np.arange(0, 1.01, 0.05)
        num_seed = 1000
        num_data = 1000
        num_class = 2
        topk = 1
        num_bins_dim = 50
        PATH = './result/simulation_setting1/'

    if setting == "simu2":
        beta_list = np.arange(0, 1.01, 0.05)
        num_seed = 1000
        num_data = 1000
        num_class = 2
        topk = 1
        num_bins_dim = 50
        PATH = './result/simulation_setting2/'

    if setting == "simu3":
        beta_list = np.arange(0, 0.1, 0.005)
        num_seed = 1000
        num_data = 1000
        num_class = 10
        topk = 2
        num_bins_dim = 20
        PATH = './result/simulation_setting3/'
        
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise
    return num_seed, num_data, num_class, topk, num_bins_dim, beta_list, PATH
    


def generate_simulation_data(setting, seed, num_data, num_class, topk, beta):
    np.random.seed(seed)
    if setting == "simu1":
        # generate Z uniformly on \Delta_{K - 1}
        Z = np.random.exponential(scale=1.0, size = [num_data, num_class])
        Z = (Z.transpose() / Z.sum(1)).transpose()
        # generate mis-calibrated model by logit transformation
        beta0 = 1
        p = Z[:, 0]
        exponent = beta0 + beta * np.log(p / (1-p))
        q = np.exp(exponent) / (1 + np.exp(exponent))
        Y_generate_prob = (Z.transpose() / (1 - p) * (1 - q)).transpose()
        Y_generate_prob[:, 0] = q
            
    if setting == "simu2":
        # generate Z with one entry follows a beta distribution
        top1_prob = np.random.beta(a = 5, b = 0.5, size=[num_data])
        other_prob = np.random.exponential(scale=1.0, size = [num_data, num_class - 1])
        other_prob = ((other_prob.transpose() / other_prob.sum(1)) * (1 - top1_prob)).transpose()
        top1_prob_label = np.random.randint(low=0, high=num_class, size=[num_data])
        Z = np.zeros([num_data, num_class])
        Z[:, 1:] = other_prob
        Z[:, 0] = top1_prob
        Z[:, 0] = Z[(np.arange(num_data), top1_prob_label)]
        Z[(np.arange(num_data), top1_prob_label)] = top1_prob
        # generate mis-calibrated model by logit transformation
        beta0 = 1
        p = Z[:, 0]
        exponent = beta0 + beta * np.log(p / (1-p))
        q = np.exp(exponent) / (1 + np.exp(exponent))
        Y_generate_prob = (Z.transpose() / (1 - p) * (1 - q)).transpose()
        Y_generate_prob[:, 0] = q
    if setting == "simu3":
        # generate Z uniformly on \Delta_{K - 1}
        Z = np.random.exponential(scale=1.0, size = [num_data, num_class])
        Z = (Z.transpose() / Z.sum(1)).transpose()
        # generate mis-calibrated model by perturbation
        Y_generate_prob = np.copy(Z)
        Z_order = np.argsort(Z)
        temp_index = np.arange(0, num_data, 1)
        Y_generate_prob[(temp_index, Z_order[:, -1])] -= beta
        Y_generate_prob[(temp_index, Z_order[:, -2])] += beta
        
    y_generate_uniform = np.random.uniform(low=0.0, high=1.0, size = [num_data])
    Y_generate_prob_cum_sum = np.cumsum(Y_generate_prob, axis = 1)
    label = (Y_generate_prob_cum_sum.transpose() < y_generate_uniform).sum(0)
    Y = np.zeros([num_data, num_class])
    Y[(np.arange(num_data), label)] = 1
    # get topk entries
    if topk == num_class:
        Y_topk = Y
        Z_topk = Z
    else:
        row_index = np.array([[row for _ in range(num_class)] for row in range(num_data)])
        Z_order = np.argsort(Z)
        Y = Y[(row_index, Z_order)]
        Z = Z[(row_index, Z_order)]
        Y_topk = Y[:, (-topk):]
        Z_topk = Z[:, (-topk):]

    return Z_topk, Y_topk
        
        
    