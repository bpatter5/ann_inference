# =============================================================================
# Created By: bpatter5 
# Updated By: bpatter5
# Created On: 12/2/2018
# Updated On: 12/2/2018
# Purpose: Simple feedforward network using PyTorch for regression problems 
# =============================================================================

from torch import nn

# Class for simple feedforward regression analysis
# inherits from nn.Module class
class RegressionFF(nn.Module):
    '''
    Description
    -----------
    Class for simple feedforward neural nets for regression problems
    
    Parameters
    ----------
    input_size : int
        number of columns in the input
    
    hidden_units : int, 10
        number of units in each hidden layer
    
    activ_func: nn.xxxx , nn.ReLU
    
    '''
    # init function takes required input_size (int) 
    # optional hidden_units (int, 10) and activ_func (PyTorch activation funct, nn.ReLU)
    def __init__(self, input_size, hidden_units=10, activ_func=nn.ReLU):
        # init superclass and linear units
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_units)
        self.activ_func = activ_func()
        self.linear_2 = nn.Linear(hidden_units, 1)
    
    # method for forward step
    # takes a PyTorch tensor as input
    # outputs a point estimate for the tensor
    def forward(self, x):
        '''
        Description
        -----------
        Function to perform forward prop through the network
        
        Parameters
        ----------
        x : PyTorch.tensor array
            input to push through on the forward pass through the network
            
        Returns
        -------
        x : PyTorch.tensor
            results of feeding x through the network
            
        '''        
        # linear forward steps with activ function
        x = self.linear_1(x)
        x = self.activ_func(x)
        x = self.linear_2(x)
        
        # return an estimate for x 
        return(x)

# function to initalize weights of a model
# takes a model and weights as input
def init_weights(m):
    '''
    Description
    -----------
    Function to reinitialize weights
    
    Parameters
    ----------
    m : torch.nn.Module
        model to reset weights on
    
    Returns
    -------
        : void
        resets weight of model m using uniform xavier init
        
    '''
    # only work on linear layers
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)
        
        