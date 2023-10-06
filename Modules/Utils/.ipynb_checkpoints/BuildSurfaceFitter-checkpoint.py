import torch, pdb
import torch.nn as nn

class BuildSurfaceFitter(nn.Module):
   
    '''
    Builds a custom multilayer perceptron (MLP) for surface fitting. This
    class allows the user to specify multiple activation functions per
    layer. Note, the number of neurons per layer will be equal to the layer
    size multiplied by the number of activations for that layer.
   
    Args:
        input_variables:   integer number of input variables
        hidden_layers:     list of integer hidden layer sizes
        outuput_variables: integer number of output variables
        activations:       list of instantiated activation functions
        linear_output:     boolean indicator for linear output
        linear_activation: instantiated activation function
   
    Inputs:
        x: torch float tensor of input variables
   
    Returns:
        y: torch float tensor of output variables
    '''
   
    def __init__(self, 
                 input_variables,
                 hidden_layers,
                 output_variables,
                 activations=None, 
                 linear_output=True,
                 output_activation=None):
       
        #
        # initialization
        #
        
        super().__init__()
        self.input_variables = input_variables
        self.hidden_layers = hidden_layers
        self.output_variables = output_variables
        self.linear_output = linear_output
        self.output_activation = output_activation
        
        # nonlinear activations
        activations = activations if activations is not None else nn.Softplus()
        if not isinstance(activations, list):
            self.activations = [activations] 
        else:
            self.activations = activations
        
        if not self.linear_output:
            assert self.output_activation is not None
        
        #
        # instantiataion
        #
        
        # linear activations
        self.linear_activations = []
        for i, layer in enumerate(self.hidden_layers):
            self.linear_activations.append(nn.Linear(
                in_features=self.input_variables,
                out_features=len(self.activations)*layer,
                bias=True))
            self.input_variables = len(self.activations)*layer
        
        # linear output layer
        self.output_layer = nn.Linear(
            in_features=self.input_variables,
            out_features=output_variables,
            bias=True)
        
        # nonlinear output
        if not self.linear_output:
            self.output_layer = nn.Sequential(
                self.output_layer,
                self.output_activation)
       
    def forward(self, x):
       
        # loop through hidden layers
        for i, linear in enumerate(self.linear_activations):
            
            # linear activation
            x = linear(x)
            
            # slice neurons for each activation function
            n = self.hidden_layers[i] # number of neurons per activation
            a = len(self.activations) # number of activations
            x = [x[:, j*n:(j+1)*n] for j in range(a)]
            print(x)
            # apply activations to slices
            x = [self.activations[j](x[j]) for j in range(a)]
            print(x)
            None[0]
            x = torch.cat(x, dim=1)
            
        # apply output layer
        y = self.output_layer(x)
       
        return y
