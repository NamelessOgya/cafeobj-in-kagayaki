
from custom_policy import CustomPolicy, PolicyMapper

import functools
import random
from copy import copy, deepcopy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

## torch学習部分
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn

## ray部分
import ray
from ray.rllib.algorithms.impala import ImpalaConfig, Impala
from ray import air
from ray import tune

from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import CSVLoggerCallback
from ray.tune.registry import register_env, get_trainable_cls
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import collections
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
import os

from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from collections import defaultdict

import yaml

with open("training_config.yaml", 'r') as file:
    training_config = yaml.safe_load(file)    




policy_mapper = PolicyMapper()


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self, policy_mapper,win_rate_threshold, model_save_freq, callback_path):
        super().__init__()
        self._matching_stats = defaultdict(int)
        self.my_callback_dir = callback_path
        self.model_save_freq = model_save_freq
        self.policy_mapper = policy_mapper
        self.win_rate_threshold = win_rate_threshold
    # def on_worker_start(self, worker, **kwargs):
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     src_path = os.path.join(current_dir, 'src')
    #     sys.path.append(src_path)

    def save_model_checkpoint(self, algorithm):
        policy = algorithm.get_policy("main")
        model = policy.model
        model_state_dict = model.state_dict()

        os.makedirs(self.my_callback_dir, exist_ok=True)
        save_path = os.path.join(self.my_callback_dir, f"main_v{self.policy_mapper.current_opponent}.pt")

        torch.save(model_state_dict, save_path)

    def on_train_result(self, *, algorithm, result, **kwargs):
        
        # 勝率計算
        policy_main_reward = result["env_runners"]["hist_stats"]["policy_main_reward"]
        win_rate = sum(1 for num in policy_main_reward if num > 0) / len(policy_main_reward )
        result["win_rate"] = win_rate
        result["model_generation"] = self.policy_mapper.current_opponent #モデルの世代も記録

        if win_rate > self.win_rate_threshold:
            self.policy_mapper.current_opponent += 1
            new_pol_id = f"main_v{self.policy_mapper.current_opponent}"

            print(f"adding new opponent to the mix ({new_pol_id}).")


            # policy_clsをそのまま渡す実装はバグがある模様。(テスト部分でバグる。)  
            # 
            main_policy = algorithm.get_policy("main")
            main_state = main_policy.get_state()
            main_config = main_policy.config
            observation_space = main_policy.observation_space
            action_space = main_policy.action_space

            new_policy = algorithm.add_policy(
                policy_id=new_pol_id,
                # policy_cls=type(main_policy),
                policy = main_policy,
                policies_to_train = ["main"],
                observation_space = observation_space,
                action_space = action_space,
                config = main_config,
                policy_state = main_state,
                policy_mapping_fn = self.policy_mapper
            )

            if self.policy_mapper.current_opponent % self.model_save_freq == 0:
              self.save_model_checkpoint(algorithm)
        else:
            print("Not good enough; will keep learning ...")

        result["league_size"] = self.policy_mapper.current_opponent + 2
        
def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
    opponent = "random"

    if hash(episode.episode_id) % 2:
        if agent_id == "stabilizer":
            return "main"
        else:
            return opponent
    else:
        if agent_id == "stabilizer":
            return opponent
        else:
            return "main"