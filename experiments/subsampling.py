import pickle
import argparse

import numpy as np

from data import get_settings, generate_simulation_data
from estimator import T_mn_estimator

parser = argparse.ArgumentParser(description='simulation')
parser.add_argument('--setting', default='simu1', type=str, help='Simulation Setting')
parser.add_argument('--num_subsample', default=500, type=int, help='Number of Subsamples')
args = parser.parse_args()


def main():
    setting = args.setting
    num_subsample = args.num_subsample
    num_seed, num_data, num_class, topk, num_bins_dim, beta_list, PATH = get_settings(setting)
    
    for beta in beta_list:
        subsample_statistic_list = np.zeros([num_seed, num_subsample])
        for seed in range(num_seed):
            Z_topk_orig, Y_topk_orig = generate_simulation_data(setting, seed, num_data, num_class, topk, beta)
            data_index = np.arange(num_data)
            subample_size = int(np.floor(np.sqrt(num_data)))
            
            for subsample_seed in range(num_subsample):
                subsample_index = np.random.choice(data_index, size=subample_size, replace = True)

                Y_topk = Y_topk_orig[subsample_index,:]
                Z_topk = Z_topk_orig[subsample_index,:]
                
                clt_statistic, _ = T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk)
                
                subsample_statistic_list[seed, subsample_seed] = clt_statistic

            
            if seed % 50 == 0 or seed == num_seed - 1:
                print("seed: {}".format(seed))

                filename = "{}/subsample_setting_{}_data_{}_class_{}_topk_{}_bin_{}_beta_{:.3f}.p".format(PATH, setting, num_data, num_class, topk, num_bins_dim, beta)
                f = open(filename, 'wb')
                pickle.dump([subsample_statistic_list], f)
                f.close()
        
    
    
if __name__ == '__main__':
    main()
