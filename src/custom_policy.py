import functools
import random
from copy import copy, deepcopy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
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

# customのselfplay callback
from collections import defaultdict

import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
import shutil
import os
from functools import partial

class CustomPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        # モデルの作成
        model_config = config["model"]
        num_outputs = action_space.n  # アクション数を取得
        model = CustomTorchModel(
            observation_space, action_space, num_outputs, model_config, name="CustomTorchModel"
        )
        # 親クラスの初期化 (action_distribution_class に TorchCategorical を設定)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            model=model,
            action_distribution_class=TorchCategorical  # アクション分布クラスを指定
        )

# 共有可能なオブジェクトとして、policy_mapping_fnを定義
class PolicyMapper:
    def __init__(self):
        self.current_opponent = 0

    def __call__(self, agent_id, episode, **kwargs):
        opponent = random.choice(
            ["random"] + [f"main_v{i}" for i in range(1, self.current_opponent + 1)]
        )
        # print(f'episode {episode.episode_id} opponent {opponent} choice = {["random"] + [f"main_v{i}" for i in range(1, self.current_opponent + 1)]}')
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



