import numpy as np


def T_mn_estimator(Y_topk, Z_topk, num_bins_dim, topk):
    num_data = Z_topk.shape[0]
    num_bins = num_bins_dim ** topk
    
    # get non-empty bins
    upper_bounds = np.zeros([num_bins, topk])
    lower_bounds = np.zeros([num_bins, topk])
    bin_count = np.zeros(num_bins)
    for bin_index in range(num_bins):
        dim = topk - 1
        temp_index = bin_index
        while dim >= 0:
            dim_index = temp_index % num_bins_dim
            upper_bounds[bin_index, dim] = (dim_index + 1) / num_bins_dim
            lower_bounds[bin_index, dim] = dim_index / num_bins_dim
            temp_index = temp_index // num_bins_dim
            dim = dim - 1
        bin_count[bin_index] = (((Z_topk >= lower_bounds[bin_index, ]) & (Z_topk < upper_bounds[bin_index, ])).sum(1) == topk).sum()
    num_nonempty_bins = (bin_count > 0).sum()
    non_empty_upper_bounds = upper_bounds[(bin_count > 0)]
    non_empty_lower_bounds = lower_bounds[(bin_count > 0)]
    non_empty_bin_count = bin_count[(bin_count > 0)]
    

    bin_estimator = np.zeros([num_nonempty_bins])
    bin_var_first_term_estimator = np.zeros([num_nonempty_bins])
    bin_var_second_term_estimator = np.zeros([num_nonempty_bins])
    for bin_index in range(num_nonempty_bins):
        bin_flag = (((Z_topk >= non_empty_lower_bounds[bin_index, ]) & (Z_topk < non_empty_upper_bounds[bin_index, ])).sum(1) == topk)
        bin_Z = Z_topk[bin_flag]
        bin_Y = Y_topk[bin_flag]
        bin_U = bin_Y - bin_Z
        
        # estimator
        bin_estimator[bin_index] = non_empty_bin_count[bin_index] * np.power(bin_U.mean(0), 2).sum() - np.power(bin_U, 2).sum(1).mean()
        if non_empty_bin_count[bin_index] > 1:
            bin_estimator[bin_index] = bin_estimator[bin_index] * non_empty_bin_count[bin_index] / (non_empty_bin_count[bin_index] - 1)
        
        # variance estimator
        bin_U_sum = bin_U.sum(0)
        bin_U_square_mean = np.matmul(bin_U.transpose(), bin_U) / non_empty_bin_count[bin_index]
        bin_U_mean = bin_U.mean(0, keepdims=True)
        bin_var_first_term_estimator[bin_index] = 4 * np.matmul(np.matmul(bin_U_mean, bin_U_square_mean), bin_U_mean.transpose() ) - 3 * np.power(np.power(bin_U_mean, 2).sum(), 2)
        bin_var_second_term_estimator[bin_index] = np.power(bin_U_mean, 2).sum()

    var_first_term = (bin_var_first_term_estimator * non_empty_bin_count / num_data).sum()
    var_second_term = np.power((bin_var_second_term_estimator * non_empty_bin_count / num_data).sum(), 2)
    clt_sigma_1 = var_first_term - var_second_term 


    clt_statistic = bin_estimator.sum() / num_data

    return clt_statistic, clt_sigma_1
