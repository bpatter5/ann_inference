# =============================================================================
# Created By: bpatter5
# Updated By: bpatter5
# Created On: 11/30/2018
# Updated On: 11/30/2018
# Purpose: Methods to generate bootstrap samples of params/outputs
# =============================================================================

import numpy as np

# helper function to sample from numpy array with replacement
# takes a numpy array and optional size as input
# outputs a np.array([size])
def boot_sample(np_array, size=599):
    return(np.random.choice(np_array, size=size))

# calculates bootstrap estimate of a given test stat
# takes a numpy array as input and optional int n_iter and numpy function
# returns a np.array([n_iter]) of bootstrap test_stats
def boot_stat(np_array, n_iter=599, test_stat=np.var):
    # preallocate space for test stats
    test_vals = np.zeros(shape=n_iter)
    
    # iterate over all test_vals
    for i in np.arange(0, n_iter):
        # calc test_stat on boot_sample
        test_vals[i] = test_stat(boot_sample(np_array))
    
    # return array of test_vals
    return(test_vals)