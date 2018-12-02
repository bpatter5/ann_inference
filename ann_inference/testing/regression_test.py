# =============================================================================
# Created By: bpatter5 
# Updated By: bpatter5
# Created On: 12/2/2018
# Updated On: 12/2/2018
# Purpose: Class file for testing regression problems
# =============================================================================

import pickle 
from ann_inference.regression.feedforward import RegressionFF as rff

# method to pickle test class
# takes an object and file as input
# outputs a pickled class object
def save_test(reg_test, file):
    # open a file in write binary mode
    with open(file, 'wb'):
        # pickle object to file
        pickle.dump(reg_test, file)

# method to load pickled file
# takes a file path as input
# returns a pickled object into memory
def load_test(file):
    # open a pickled file in binary mode
    with open(file, 'rb'):
        # return the pickled object
        return(pickle.load(file))
    
# class to run testing on FF networks
# class takes a state dict and model data as input
class RegressionTester():
    # method takes a PyTorch state_dict and 
    # ann_inference.data.load_data.ModelData as input
    def __init__(self, state_dict, model_data):
        self.model = rff()
        self.model.load_state_dict(state_dict)
        self.model_data = model_data
        
    