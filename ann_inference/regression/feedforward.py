# =============================================================================
# Created By: bpatter5 
# Updated By: bpatter5
# Created On: 12/2/2018
# Updated On: 12/2/2018
# Purpose: Simple feedforward network using PyTorch for regression problems 
# =============================================================================

from torch import nn
from ann_inference.data.load_data import ModelData

class RegressionFF(nn.Sequential):
    def __init__(self):
        super().__init__()
        