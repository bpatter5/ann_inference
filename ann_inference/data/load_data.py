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
def gen_regression(samples, features, inform, bias=0.0, noise=1.0, random_state=None):
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
    else:
        return(None)

# Class for holding data for model training and testing
class ModelData():
    # init takes as input an X (np.array[n x d]), y (np.array[n]), seed (None or int/float)
    # train_pct (float)
    def __init__(self, X, y, seed, train_pct):
        self.X = X
        self.y = y
        self.seed = seed
        self.train_pct = train_pct
        np.random.seed(seed=seed)
        
        self.train_idx, self.test_idx = self.train_test_indices()
    
    # method to split train and test indices    
    def train_test_indices(self):
        # store index values in np.array
        index = np.arange(0, self.y.shape[0])
        
        # generate train and test indices
        train_idx = np.random.choice(index, size= int(self.train_pct * index.shape[0]), replace=False)
        test_idx = index[~np.in1d(index, train_idx)]
        
        # return tuple of train and test indices
        return((train_idx, test_idx))
    
    # reset seed on random number generator
    def reset_seed(self, seed):
        np.random.seed(seed=seed)
    
    # method to get training samples from X
    def get_X_train(self):
        return(self.X[self.train_idx,:])
    
    # method to get training samples from y
    def get_y_train(self):
        return(self.y[self.train_idx])
    
    # method to get test samples from X
    def get_X_test(self):
        return(self.X[self.test_idx,:])
    
    # method to get test samples from y
    def get_y_test(self):
        return(self.y[self.test_idx])
    
    # method to set mean(y_train) = 0 
    def center_y(self):
        return(self.get_y_test() - np.mean(self.get_y_test()))
    
    # method to get centered and scaled X_train
    def center_scale_X(self):
        X_train = self.get_X_train()
        
        # set mean of X_train to 0
        X_train = X_train - np.mean(X_train, axis = 1)
        
        # scale X_train to have unit std deviation along columns
        return(X_train / np.std(X_train, axis = 1))
    
    # method to get column means from X_train
    def get_col_means_X(self):
        return(np.mean(self.get_X_train(), axis = 1))
    
    # method to get column std dev from mean 0 X_train
    def get_col_scale_X(self):
        X_train = self.get_X_train() - np.mean(self.get_X_train(), axis = 1)
        
        return(np.std(X_train, axis = 1))
    
    # method to get mean of y_train 
    def get_mean_y(self):
        return(np.mean(self.get_y_train()))
    
    # method to split training index into batches
    # takes the number of batches (int) as input
    # outputs breaks in the index
    def split_batches(self, num_batches):
        # list to hold batches
        batches = list()
        # calc the batch size to generate num_batches
        batch_size = int(self.train_idx.shape[0] / num_batches)
        # calc the breaks in the index
        breaks = np.linspace(0, self.train_idx.shape[0], num = batch_size, dtype=int)
        
        # iterate over breaks
        for i in np.arange(0, breaks.shape[0] - 1):
            # assign index values to list element
            batches.append(self.train_idx[breaks[i]:breaks[i + 1]])
        
        # return list[np.array] of batches
        return(batches)
    
    # method to shuffle the training index    
    def shuffle_train_idx(self):
        # shuffle the training index
        np.random.shuffle(self.train_idx)
            
        