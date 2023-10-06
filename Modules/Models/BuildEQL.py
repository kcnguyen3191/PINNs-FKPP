import torch, pdb
import torch.nn as nn

from Modules.Layers.EQLActivationLayer import EQLActivationLayer

class BuildSurfaceFitter(nn.Module):
   
    '''
    Builds an equation learning (EQL) network. The number of neurons per
    hidden layer are determined by the provided unary and binary lists of
    activation functions (where unary implies 1 input / 1 output and binary
    implies 2 inputs / 1 output).
   
    Args:
        inputs    (int): integer number of input variables
        layers    (int): integer number of hidden layers
        outuputs  (int): integer number of output variables
        unaries  (list): list of instantiated unary activation functions
        binaries (list): list of instantiated binary activation functions
   
    Inputs:
        x (float tensor): input variables (e.g. u, u_x, u_xx, etc.)
   
    Returns:
        y (float tensor): predicted output variables (e.g. u_t)
    '''
   
    def __init__(self, inputs, layers, outuputs, unaries=[], binaries=[]):
        
        super().__init__()
        self.inputs = inputs
        self.layers = layers
        self.outuputs = outuputs
        self.unaries = unaries
        self.binaries = binaries
        
        # if no activations, just use linear combination
        if len(self.unaries) == 0 and len(self.binaries) == 0:
            self.layers = 1
            self.unaries = [nn.Identity()]
        
        # build EQL
        layers = []
        for i in range(self.layers):
            layers.append(EQLActivationLayer(
                inputs=self.inputs, 
                unaries=self.unaries, 
                binaries=self.binaries))
            self.inputs = len(self.unaries) + len(self.binaries)
        
        # linear output layer
        self.output_layer = nn.Linear(
            in_features=self.inputs,
            out_features=self.outputs,
            bias=True)
            
        # combine into sequential model
        layers.append(self.output_layer)
        self.EQL = nn.Sequential(*layers)
       
    def forward(self, x):
       
        # run the model
        y = self.EQL(x)
       
        return y
