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
        self._tanh1 = nn.Tanh()
        self._ll2 = nn.Linear(neurons[0], output_dim)
        self._sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        """
        Args:
            x (Tensor) : attributes
        """
        # for layer in self._layers:
        #     x = layer(x)

        x = self._ll1(x)
        x = self._tanh1(x)
        x = self._ll2(x)
        x = self._sigmoid1(x)

        return x