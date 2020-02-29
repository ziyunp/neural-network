import numpy as np

import torch
import torch.nn as nn

class ClaimNet(nn.Module):

    def __init__(self, input_dim, output_dim, neurons, activations):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimNet, self).__init__()
        self._ll1 = nn.Linear(input_dim, neurons[0])
        self._relu1 = nn.ReLU()
        self._ll2 = nn.Linear(neurons[0], neurons[1])
        self._relu2 = nn.ReLU()
        self._ll3 = nn.Linear(neurons[1], output_dim)
        self._sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        """
        Args:
            x (Tensor) : attributes
        """
        x = self._ll1(x)
        x = self._relu1(x)
        x = self._ll2(x)
        x = self._relu2(x)
        x = self._ll3(x)
        x = self._sigmoid1(x)

        return x