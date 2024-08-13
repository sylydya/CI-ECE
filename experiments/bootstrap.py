import os
import pickle
import argparse

import numpy as np

from data import get_settings, generate_simulation_data
from estimator import T_mn_estimator

parser = argparse.ArgumentParser(description='simulation')
parser.add_argument('--setting', default='simu1', type=str, help='Simulation Setting')
parser.add_argument('--num_bootstrap', default=200, type=int, help='Number of Bootstrap Samples')
args = parser.parse_args()


def main():
    setting = args.setting
    num_bootstrap = args.num_bootstrap
    num_seed, num_data, num_class, topk, num_bins_dim, beta_list, PATH = get_settings(setting)
    
    for beta in beta_list:
        bootstrap_statistic_list = np.zeros([num_seed, num_bootstrap])
        for seed in range(num_seed):
            Z_topk_orig, Y_topk_orig = generate_simulation_data(setting, seed, num_data, num_class, topk, beta)
            data_index = np.arange(num_data)
            
            for bootstrap_seed in range(num_bootstrap):
                bootstrap_index = np.random.choice(data_index, size=num_data, replace = True)

                Y_topk = Y_topk_orig[bootstrap_index,:]
                Z_topk = Z_topk_orig[bootstrap_index,:]
                
                clt_statistic, _ = T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk)
                
                bootstrap_statistic_list[seed, bootstrap_seed] = clt_statistic

            
            if seed % 50 == 0 or seed == num_seed - 1:
                print("seed: {}".format(seed))

                filename = "{}/bootstrap_setting_{}_data_{}_class_{}_topk_{}_bin_{}_beta_{:.3f}.p".format(PATH, setting, num_data, num_class, topk, num_bins_dim, beta)
                f = open(filename, 'wb')
                pickle.dump([bootstrap_statistic_list], f)
                f.close()
        
    
    
if __name__ == '__main__':
    main()
