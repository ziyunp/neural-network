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

        # Extremely difficult to implement this
        # self._layers = []
        # n_inputs = input_dim
        # for i in range(len(neurons)):
        #     self._layers.append(nn.Linear(n_inputs, neurons[i]))
        #     if activations[i] == "relu":
        #         self._layers.append(nn.ReLU())
        #     elif activations[i] == "sigmoid":
        #         self._layers.append(nn.Sigmoid())
        #     elif activations[i] == "softmax":
        #         self._layers.append(nn.Softmax(dim=1))
        #     elif activations[i] == "tanh":
        #         self._layers.append(nn.Tanh())
        #     n_inputs = neurons[i]  

        self._ll1 = nn.Linear(input_dim, neurons[0])
        self._ll2 = nn.Linear(neurons[0], neurons[1])
        self._ll3 = nn.Linear(neurons[1], neurons[2])


    def forward(self, x):
        """
        Args:
            x (Tensor) : attributes
        """
        # for layer in self._layers:
        #     x = layer(x)

        x = F.relu(self._ll1(x))
        x = F.relu(self._ll2(x))
        x = torch.sigmoid(self._ll3(x))

        return x