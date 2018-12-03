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
        
        # linear forward steps with activ function
        x = self.linear_1(x)
        x = self.activ_func(x)
        x = self.linear_2(x)
        
        # return an estimate for x 
        return(x)

# function to initalize weights of a model
# takes a model and weights as input
def init_weights(m, init_func):
    # only work on linear layers
    if type(m) == nn.Linear:
        init_func(m.weight)
        m.bias.data.fill_(0.01)
        
        