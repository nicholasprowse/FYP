
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm


class MLP(nn.Module):
    """
    Multilayer Perceptron Model. This is a two layer fully connected multilayer perceptron. Both layers
    have configurable dropout rate. There is an activation function between layers (relu or gelu).

    The input and output are both determined by hidden size, while the hidden layer has size determined by
    mlp_dim in the config.
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = f.relu        # TODO: Test this with both relu and gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        # Biases initialised using normal distribution, weights initialised using xavier uniform,
        # which is a symmetric uniform distribution with the range inversely proportional to the
        # sum of the fan in and fan out (input and output weights).
        # TODO Investigate why xavier uniform is beneficial and try alternate initialisations
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x