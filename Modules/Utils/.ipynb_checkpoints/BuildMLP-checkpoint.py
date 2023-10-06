import torch, pdb
import torch.nn as nn

class BuildMLP(nn.Module):
    
    '''
    Builds a standard multilayer perceptron (MLP) with options.
    
    Args:
        input_features: integer number of input features
        layers:         list of integer layer sizes
        activation:     instantiated activation function
        linear_output:  boolean indicator for linear output
    
    Inputs:
        x: torch float tensor of inputs
    
    Returns:
        y: torch float tensor of outputs
    '''
    
    def __init__(self, input_features, layers, activation=None, linear_output=True):
        
        # initialization
        super().__init__()
        self.input_features = input_features
        self.layers = layers
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.linear_output = linear_output
        
        # instantiataion
        operations = []
        for i, layer in enumerate(layers):
            
            # add linear activation
            operations.append(nn.Linear(
                in_features=self.input_features,
                out_features=layer,
                bias=True))
            self.input_features = layer
            
            # add nonlinear activation 
            if i < len(self.layers)-1 or self.linear_output:
                operations.append(self.activation)
                
        # convert module list to sequential model
        self.MLP = nn.Sequential(*operations)
        
    def forward(self, x):
        
        # run the model
        y = self.MLP(x)
        
        return y
        
        