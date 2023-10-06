import torch
import numpy as np

def SELUInitialization(model):
    
    '''
    Loop over MLP model parameters, zero initialize if 1D tensor (i.e. bias) 
    and SELU initialize by drawing from N(0, 1/input_size) if nD tensor 
    (i.e. weight).
    '''

    # loop over model parameters
    for tensor in list(model.parameters()):

        # zero initialize bias vectors
        if len(tensor.shape) == 1:
            tensor.data = torch.zeros_like(tensor.data)
        
        # selu initialize weight matrices
        elif len(tensor.shape) == 2:
            size = list(tensor.data.size())
            std = 1 / size[0]
            weight = np.random.normal(size=size) * std
            tensor.data = torch.from_numpy(weight).float().to(tensor.data.device)
       
        # else raise error
        else:
            raise ValueError('Encountered tensor with more than 2 dimensions.')
