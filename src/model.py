import functools
import random
from copy import copy, deepcopy

import numpy as np

## torch学習部分
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn

## ray部分
import ray
import collections

import os





class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # print(f'num_outputs = {num_outputs}')
        # assert num_outputs == 2, f'Assertion failed: num_outputs must be 2 but num_outputs={num_outputs}'
        self.num_outputs = num_outputs
        self._num_objects = obs_space.shape[0]
        self._num_actions = num_outputs

        hidden_depth = model_config["custom_model_config"].get("config_hidden_depth")
        hidden_dim = model_config["custom_model_config"].get("config_hidden_dim")
        
        # hidden_depth = 8
        # hidden_dim = 128

        layers = [nn.Linear(self._num_objects, hidden_dim), nn.ReLU()]
        for i in range(hidden_depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        # value function
        self.vf_head = nn.Linear(hidden_dim, 1)

        # action logits（ソフトマックスは適用しない）
        self.ac_head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        # assert not isinstance(x, collections.OrderedDict) , f'input is orderdict {x}'
        x = self.layers(x)
        logits = self.ac_head(x)
        self.value_out = self.vf_head(x)
        return logits, []

    def value_function(self):
        return torch.reshape(self.value_out, (-1,))

    

