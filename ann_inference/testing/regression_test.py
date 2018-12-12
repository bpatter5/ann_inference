# =============================================================================
# Created By: bpatter5 
# Updated By: bpatter5
# Created On: 12/2/2018
# Updated On: 12/2/2018
# Purpose: Class file for testing regression problems
# =============================================================================

import numpy as np
import pyarrow as pa
from ann_inference.data import load_data as ld
from ann_inference.regression import feedforward  as rff
from ann_inference.data import arrow_helper as ah
import torch
from torch.nn import functional as F
import pandas as pd
from pyarrow import parquet as pq


# class to run testing on FF networks
# class takes a state dict and model data as input
class RegressionTester():
    '''
    Description
    -----------
    Class to run testing on PyTorch models. Relies heavily on the Apache Arrow project 
    and the Parquet format to efficiently write model parameters to disk following each run of fit to avoid 
    overloading in-memory operations. The Parquet format allows for the files on disk to be read and analyzed by a variety of tools
    {Spark, Drill, Pandas, etc.}

    Parameters
    ----------
    model_data : ann_inference.data.load_data.ModelData
        Test data to input into model. Using synthetic so all aspects of the dataset are
        known and controlled to make for easy testing across a variety of parameters.
    
    hidden_units : int
        Desired number of hidden units per hidden layer in the PyTorch model
    
    opt : torch.optim.xxxx, torch.optim.SGD
    
    loss_func : torch.nn.Functional.xxxx, torch.nn.Functional.mse_loss
    
    '''
    def __init__(self, model_data, hidden_units, opt = torch.optim.SGD, loss_func = F.mse_loss):
        self.model_data = model_data
        self.hidden_units = hidden_units
        self.model = rff.RegressionFF(input_size = self.model_data.X.shape[1], hidden_units=hidden_units)
        self.opt = opt(self.model.parameters(), 1e-5)
        self.loss_func = loss_func
        
    # method to fit model to dataset 
    def fit(self, num_epochs, num_batch, test_id, path):
        '''
        
        Description
        -----------
        Function to fit model for a given number of epochs, batches, ids, and path.
        
        Parameters
        ----------
        num_epochs : int
            number of training epochs
        
        num_batch : int
            number of batches for the model fit
        
        test_id : int
            id for the current test iteration
        
        path : string
            path to write parquet datasets to for later analysis
        
        Returns
        -------
            : void
            writes parameters and loss function to disk
            
        '''
        # zero grads 
        self.opt.zero_grad()
        
        # generate torch tensors from numpy arrays in model_data
        X = torch.Tensor(self.model_data.X)
        y = torch.Tensor(np.reshape(self.model_data.y, [self.model_data.y.shape[0] , 1]))
        
        
        # storing mse for each epoch 
        batch_dict = {'weight_1':list(), 'weight_2':list()}
        
        weight_cols = {'mse':['stat', 'id', 'epoch_num', 'mse']}
        
        weight_cols['weight_1'] = []
        weight_cols['weight_1'].append('stat')
        weight_cols['weight_1'].append('id')
        weight_cols['weight_1'].append('epoch_num')
        
        weight_cols['weight_1'].extend(['weight_' + str(i) for i in np.arange(0, self.model.linear_1.weight.shape[0])])
        
        weight_cols['weight_2'] = []
        weight_cols['weight_2'].append('stat')
        weight_cols['weight_2'].append('id')
        weight_cols['weight_2'].append('epoch_num')
        
        weight_cols['weight_2'].extend(['weight_' + str(i) for i in np.arange(0, self.model.linear_2.weight.shape[0])])
        
        mse_epoch = np.zeros([num_epochs])
        #weight_1 = np.zeros([num_epochs, self.model.linear_1.weight.shape[0]])
        #weight_2 = np.zeros([num_epochs, self.model.linear_2.weight.shape[0]])
        
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
            epoch_loss = self.model.forward(X[self.model_data.test_idx,:])
            
            error_calc = self.loss_func(epoch_loss, y[self.model_data.test_idx,:])
            mse_epoch[epoch] = error_calc.detach().numpy()
            
            #batch_dict['mse'].append(ah.gen_parquet_batch(float(error_calc.detach().numpy()), fill_col='mse',
                      #epoch_num=epoch, test_num=test_id, col_names=weight_cols['mse']))
            
            # storing the mean weight for each epoch along the columns
            weight1_calc = np.mean(self.model.linear_1.weight.detach().numpy(), axis=1)
           # print(weight1_calc.shape)
            weight1_calc = np.reshape(weight1_calc, [1, weight1_calc.shape[0]])
            
            batch_dict['weight_1'].append(ah.gen_parquet_batch(weight1_calc ,
                      fill_col='weight_1', epoch_num=epoch, test_num=test_id, col_names=weight_cols['weight_1']))
            
            weight2_calc = np.mean(self.model.linear_2.weight.detach().numpy(), axis=1)
           # print(weight2_calc.shape)
            weight2_calc = np.reshape(weight2_calc, [1, weight2_calc.shape[0]])
            
            batch_dict['weight_2'].append(ah.gen_parquet_batch(weight2_calc ,fill_col='weight_2' ,
                      epoch_num=epoch, test_num=test_id, col_names=weight_cols['weight_2']))
        
        ah.array_to_parquet(path=path, test_stat=mse_epoch, fill_col='mse',
                  test_num=test_id, col_names=weight_cols['mse'])
        for key, batches in batch_dict.items():
            if key != 'mse':
                ah.results_to_parquet(path=path, batch_list=batches)
            
        
    def gen_test_datasets(self, num_tests, num_epochs, num_batch, path, seed):
        '''
        Description
        -----------
        Function to write multiple model fits to disk in parquet format.
            
        Parameters
        ----------
        num_epochs : int
            number of training epochs
        
        num_batch : int
            number of batches for the model fit
        
        test_id : int
            id for the current test iteration
        
        path : string
            path to write parquet datasets to for later analysis
            
        Returns
        -------
            : void
            writes multiple results to disk in parquet format
                
        '''
        if seed is not None:
            for i in np.arange(0, num_tests):
                torch.manual_seed(seed)
                self.model.apply(rff.init_weights)
                self.fit(num_epochs=num_epochs, num_batch=num_batch, path=path, test_id=i)
        else:
            for i in np.arange(0, num_tests):
                torch.manual_seed(np.random.randint(-1000, 1000))
                self.model.apply(rff.init_weights)
                self.fit(num_epochs=num_epochs, num_batch=num_batch, path=path, test_id=i)
                
    def gen_test_datasets_permuted(self, num_tests, num_epochs, num_batch, path, seed):
        '''
        Description
        -----------
        Function to write multiple model fits to disk in parquet format.
            
        Parameters
        ----------
        num_epochs : int
            number of training epochs
        
        num_batch : int
            number of batches for the model fit
        
        test_id : int
            id for the current test iteration
        
        path : string
            path to write parquet datasets to for later analysis
            
        Returns
        -------
            : void
            writes multiple results to disk in parquet format
                
        '''
        if seed is not None:
            for i in np.arange(0, num_tests):
                torch.manual_seed(seed)
                self.model.apply(rff.init_weights)
                self.fit_permute(num_epochs=num_epochs, num_batch=num_batch, path=path, test_id=i)
        else:
            for i in np.arange(0, num_tests):
                torch.manual_seed(np.random.randint(-1000, 1000))
                self.model.apply(rff.init_weights)
                self.fit_permute(num_epochs=num_epochs, num_batch=num_batch, path=path, test_id=i)
        
    def fit_permute(self, num_epochs, num_batch, test_id, path):
        '''
        
        Description
        -----------
        Function to fit model for a given number of epochs, batches, ids, and path.
        
        Parameters
        ----------
        num_epochs : int
            number of training epochs
        
        num_batch : int
            number of batches for the model fit
        
        test_id : int
            id for the current test iteration
        
        path : string
            path to write parquet datasets to for later analysis
        
        Returns
        -------
            : void
            writes parameters and loss function to disk
            
        '''
            # zero grads 
        self.opt.zero_grad()
        
        # generate torch tensors from numpy arrays in model_data
        X = torch.Tensor(self.model_data.X)
        y = torch.Tensor(np.reshape(self.model_data.y, [self.model_data.y.shape[0] , 1]))
        
        
        # storing mse for each epoch 
        weight_cols = {'mse':['stat', 'id', 'epoch_num', 'mse'], 'permute_mse':['stat', 'id', 'epoch_num', 'mse']}
        
        
        
        mse_epoch = np.zeros([num_epochs])
        mse_perm = np.zeros([num_epochs])
        #weight_1 = np.zeros([num_epochs, self.model.linear_1.weight.shape[0]])
        #weight_2 = np.zeros([num_epochs, self.model.linear_2.weight.shape[0]])
        
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
            epoch_loss = self.model.forward(X[self.model_data.test_idx,:])
            
            error_calc = self.loss_func(epoch_loss, y[self.model_data.test_idx,:])
            mse_epoch[epoch] = error_calc.detach().numpy()
            
            
            perm_error = self.loss_func(epoch_loss, y[np.random.choice(self.model_data.test_idx, self.model_data.test_idx.shape[0], replace=False),:])
            mse_perm[epoch] = perm_error.detach().numpy()
            
        
        ah.array_to_parquet(path=path, test_stat=mse_epoch, fill_col='mse',
                  test_num=test_id, col_names=weight_cols['mse'])
        
        ah.array_to_parquet(path=path, test_stat=mse_perm, fill_col='permuted_mse',
                  test_num=test_id, col_names=weight_cols['permute_mse'])
            


def _serialize_RegressionTester(test):
    '''
    Description
    -----------
    Custom serializer for RegressionTester.
    
    Parameters
    ----------
    test : RegressionTester
        RegressionTester to serialize
    
    Returns
    -------
        : dict
        dict containing alls the elements needed to init a new RegressionTester
    '''
    return({'X':test.model_data.X, 'y':test.model_data.y, 'seed':test.model_data.seed,
            'train_pct':test.model_data.train_pct, 'hidden_units':test.hidden_units})

def _deserialize_RegressionTester(test):
    '''
    
    Description
    -----------
    Custom deserialization routine for RegressionTester.
    
    Parameters
    ----------
    test : RegressionTester
        RegressionTester to serialize
    
    Returns
    -------
        : RegressionTester
        
    '''
    return(RegressionTester(ld.ModelData(test['X'], test['y'], test['seed'],
                                          test['train_pct']), test['hidden_units']))  
    
# init SerializationContext for RegressionTester 
context = pa.SerializationContext()
context.register_type(RegressionTester, 'RegressionTester' , 
                      custom_serializer=_serialize_RegressionTester ,
                      custom_deserializer=_deserialize_RegressionTester)
