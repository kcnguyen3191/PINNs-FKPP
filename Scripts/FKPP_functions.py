import torch, pdb
import torch.nn as nn

from Modules.Models.BuildMLP import BuildMLP
from Modules.Activations.SoftplusReLU import SoftplusReLU

#
# Surface Fitters
#

# surface fitter MLP
class SurfaceFitter(nn.Module):
    def __init__(self, K=1.7e3):
        super().__init__()
        self.K = K
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[128, 128, 128, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
    def forward(self, inputs):
        return self.K*self.mlp(inputs)

#
# Diffusion Terms
#
    
# classical FKPP diffusion
class ScalarDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = 1
        self.min = 0 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.max = 4000 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.D = nn.Parameter(torch.rand(1)*(self.max - self.min) + self.min)
    def forward(self, u):
        return self.D*torch.ones_like(u)
#
# Reaction Terms
#

# classical FKPP growth
class LogisticGrowth(nn.Module):
    def __init__(self, K=1.7e3):
        super().__init__()
        self.inputs = 1
        self.K = K
        self.min = -0.02 / (1/24) # 1/hr -> 1/d
        self.max = 0.1 / (1/24) # 1/hr -> 1/d
        self.r = nn.Parameter(torch.rand(1)*(self.max - self.min) + self.min)
    def forward(self, u):
        return self.r*(1 - u/self.K)
    