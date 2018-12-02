# =============================================================================
# Created By: bpatter5
# Updated By: bpatter5
# Created On: 11/30/2018
# Updated On: 11/30/2018 
# Purpose: Gather dataset for testing methods to load linear model and linear
# models with nonlinear transformations
# =============================================================================

from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3 
import numpy as np

# dict to store sklearn dataset creation functions
datasets = {0:make_regression, 1:make_friedman1, 2:make_friedman2, 3:make_friedman3}

# function to generate linear regression dataset
# takes as required input: samples (int), features (int), inform (int)
# optional input: bias (float), noise (float), random_state (RandomState)
# returns numpy arrays of X,y  
def gen_regression(samples, features, inform, bias=np.random.normal(), noise=1.0, random_state=None):
    return(datasets[0](n_samples=samples, n_features=features, n_informative=inform, bias=bias, noise=noise, random_state=random_state))

# function to generate samples for friedman regression problems (see sklearn docs for more info)
# takes as required input: i (int), samples (int)
# optional input: noise (float), random_state (RandomState), features (int)
# returns either numpy arrays of X,y or None
def gen_friedman(i, samples, noise=1.0, random_state=None, features=20):
    if(i==1):
        return(datasets[i](n_samples=samples, n_features=features, noise=noise, random_state=random_state))
    elif(i==2 or i==3):
        return(datasets[i](n_samples=samples, noise=noise, random_state=random_state))
    elif(i==3):
        return(None)
