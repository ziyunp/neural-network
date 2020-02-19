import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClaimNet(nn.Module):

    def __init__(self, input_dim, neurons, activations):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimNet, self).__init__()

        self._layers = []
        n_inputs = input_dim
        for i in range(len(neurons)):
            self._layers.append(nn.Linear(n_inputs, neurons[i]))
            if activations[i] == "relu":
                self._layers.append(nn.ReLU())
            elif activations[i] == "sigmoid":
                self._layers.append(nn.Sigmoid())
            elif activations[i] == "softmax":
                self._layers.append(nn.Softmax(dim=1))
            elif activations[i] == "tanh":
                self._layers.append(nn.Tanh())
            n_inputs = neurons[i]  

    def forward(self, x):
        """
        Args:
            x (Tensor) : attributes
        """
        for layer in self._layers:
            x = layer(x)
        return x