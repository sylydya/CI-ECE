import pickle
import argparse

import numpy as np

from data import get_settings, generate_simulation_data
from estimator import T_mn_estimator

parser = argparse.ArgumentParser(description='simulation')
parser.add_argument('--setting', default='simu1', type=str, help='Simulation Setting')
args = parser.parse_args()

def solve_for_B(alpha=0.1, Delta=0.0, t = 0.0):
    B_low = max(np.floor(np.emath.logn(2 + 2*t, (2 + 2*t)/alpha )), np.floor(np.emath.logn((2 + 2*t)/(1 + 2*Delta), (1 + t)/alpha )))
    B_up = np.ceil(np.emath.logn((2 + 2*t)/(1 + 2*Delta), (2 + 2*t)/alpha))
    def function_Q(B):
        return ((1/2 - Delta)**B + (1/2 + Delta)**B)*(1 + t)**(-B + 1)
    for B in range(int(B_low), int(B_up) + 1):
        if function_Q(B) <= alpha:
            break
    return B


def main():
    setting = args.setting
    num_seed, num_data, num_class, topk, num_bins_dim, beta_list, PATH = get_settings(setting)
    
    for beta in beta_list:
        hulc_statistic_dict = {}
        for seed in range(num_seed):
            Z_topk_orig, Y_topk_orig = generate_simulation_data(setting, seed, num_data, num_class, topk, beta)
            full_statistics, _ = T_mn_estimator(Y_topk_orig, Z_topk_orig, num_bins_dim, topk)
            
            # Adaptive HulC
            # estimate median bias by subsampling
            num_subsample = 500
            subample_size = int(np.floor(np.sqrt(num_data)))
            subsample_statistic_list = np.zeros([num_subsample])
            data_index = np.arange(num_data)
            for subsample_seed in range(num_subsample):
                subsample_index = np.random.choice(data_index, size=subample_size,  replace = False)
                Y_topk = Y_topk_orig[subsample_index,:]
                Z_topk = Z_topk_orig[subsample_index,:]
                clt_statistic, _ = T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk)
                subsample_statistic_list[subsample_seed] = clt_statistic
            
            # compute number of split needed
            alpha = 0.1
            Delta = np.abs((subsample_statistic_list < full_statistics).mean() - 0.5)
            B1 = solve_for_B(Delta = Delta)
            B_hulc = B1
            p1 = (1/2 + Delta)**B1 + (1/2 - Delta)**B1
            B0 = B1 - 1
            p0 = (1/2 + Delta)**B0 + (1/2 - Delta)**B0
            U = np.random.uniform(0,1,1)
            tau = (alpha - p1)/(p0 - p1)
            B_hulc = B0*(U <= tau)+ B1*(U > tau)
            
            num_split = int(B_hulc)
            hul_statistic_list = np.zeros([num_split])
            hulc_split_n = num_data // num_split
            
            for hulc_seed in range(num_split):
                if hulc_seed == num_split - 1:
                    hulc_index = data_index[(hulc_seed * hulc_split_n) : ]
                else:
                    hulc_index = data_index[(hulc_seed * hulc_split_n) : ((hulc_seed + 1) * hulc_split_n)]

                Y_topk = Y_topk_orig[hulc_index,:]
                Z_topk = Z_topk_orig[hulc_index,:]
                clt_statistic, _ = T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk)
                hul_statistic_list[hulc_seed] = clt_statistic
                
            hulc_statistic_dict[seed] = hul_statistic_list

            if seed % 50 == 0 or seed == num_seed - 1:
                print("seed: {}".format(seed))
                filename = "{}/hulc_setting_{}_data_{}_class_{}_topk_{}_bin_{}_beta_{:.3f}.p".format(PATH, setting, num_data, num_class, topk, num_bins_dim, beta)
                f = open(filename, 'wb')
                pickle.dump([hulc_statistic_dict], f)
                f.close()
    
if __name__ == '__main__':
    main()
