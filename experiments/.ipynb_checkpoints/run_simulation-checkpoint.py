import os
import pickle
import argparse

import numpy as np

from data import generate_simulation_data
from estimator import T_mn_estimator

parser = argparse.ArgumentParser(description='simulation')
parser.add_argument('--setting', default='simu1', type=str, help='Simulation Setting')
args = parser.parse_args()

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
    


def main():
    setting = args.setting
    num_seed, num_data, num_class, topk, num_bins_dim, beta_list, PATH = get_settings(setting)
    
    for beta in beta_list:
        clt_statistic_list = np.zeros([num_seed])
        clt_sigma_1_list = np.zeros([num_seed])
        for seed in range(num_seed):
            Z_topk, Y_topk = generate_simulation_data(setting, seed, num_data, num_class, topk, beta)
            
            clt_statistic, clt_sigma_1 = T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk)

            clt_statistic_list[seed] = clt_statistic
            clt_sigma_1_list[seed] = clt_sigma_1
            if seed % 50 == 0 or seed == num_seed - 1:
                print("seed: {}".format(seed))
                print("clt statistics: {}".format(clt_statistic))
                print("clt single var: {}".format(clt_sigma_1))

                filename = "{}/setting_{}_data_{}_class_{}_topk_{}_bin_{}_beta_{:.3f}.p".format(PATH, setting, num_data, num_class, topk, num_bins_dim, beta)
                f = open(filename, 'wb')
                pickle.dump([clt_statistic_list, clt_sigma_1_list], f)
                f.close()
        
    
    
if __name__ == '__main__':
    main()
