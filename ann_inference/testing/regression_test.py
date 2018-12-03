# =============================================================================
# Created By: bpatter5 
# Updated By: bpatter5
# Created On: 12/2/2018
# Updated On: 12/2/2018
# Purpose: Class file for testing regression problems
# =============================================================================

import numpy as np
import pickle 
from ann_inference.regression import feedforward  as rff
import torch
from torch.nn import functional as F

# method to pickle test class
# takes an object and file as input
# outputs a pickled class object
def save_test(reg_test, file):
    # open a file in write binary mode
    with open(file, 'wb') as f:
        # pickle object to file
        pickle.dump(reg_test, f)

# method to load pickled file
# takes a file path as input
# returns a pickled object into memory
def load_test(file):
    # open a pickled file in binary mode
    with open(file, 'rb') as f:
        # return the pickled object
        return(pickle.load(f))
    
# class to run testing on FF networks
# class takes a state dict and model data as input
class RegressionTester():
    # method takes a PyTorch state_dict and 
    # ann_inference.data.load_data.ModelData as input
    def __init__(self, model_data, hidden_units, opt = torch.optim.SGD, loss_func = F.mse_loss):
        self.model_data = model_data
        self.model = rff.RegressionFF(input_size = self.model_data.X.shape[1], hidden_units=hidden_units)
        self.opt = opt(self.model.parameters(), 1e-5)
        self.loss_func = loss_func
        
    # method to fit model to dataset 
    def fit(self, num_epochs, num_batch):
        # zero grads 
        self.opt.zero_grad()
        
        # generate torch tensors from numpy arrays in model_data
        X = torch.Tensor(self.model_data.X)
        y = torch.Tensor(np.reshape(self.model_data.y, [self.model_data.y.shape[0] , 1]))
        
        # storing mse for each epoch 
        mse_epoch = np.zeros(num_epochs)
        weight_1 = np.zeros([num_epochs, self.model.linear_1.weight.shape[0]])
        weight_2 = np.zeros([num_epochs, self.model.linear_2.weight.shape[0]])
        
        # iterate over samples for num_empochs
        for epoch in range(num_epochs):
            
            # split the training index into batches
            batches = self.model_data.split_batches(num_batch)
            # iterate over every batch in batches
            for batch in batches:
                
                # generate predictions via the forward func
                pred = self.model.forward(X[batch,:])
                # calc the loss on the predict for the batch
                loss = self.loss_func(pred, y[batch,:])
                
                # clear the gradients
                loss.backward()
                # update the weights in the network
                self.opt.step()
                # zero the gradients in the network
                self.opt.zero_grad()
                # shuffle the training indices so the batches
                # are different each epoch
                self.model_data.shuffle_train_idx()
            
            # predicted loss compared to actual at epoch
            epoch_loss = self.model.forward(X[self.model_data.train_idx,:])
            mse_epoch[epoch] = self.loss_func(epoch_loss, y[self.model_data.train_idx,:])
            # storing the mean weight for each epoch along the columns
            weight_1[epoch,:] = np.mean(self.model.linear_1.weight.detach().numpy(), axis=1)
            weight_2[epoch,:] = np.mean(self.model.linear_2.weight.detach().numpy(), axis=1)
            
        return((mse_epoch, weight_1, weight_2))