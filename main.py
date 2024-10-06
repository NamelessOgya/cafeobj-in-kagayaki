import gymnasium as gym
import functools
import random
from copy import copy, deepcopy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import ParallelEnv

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

PROJ_NAME = "test_PvP_cartpole_AEC"

# 実行設定
USE_WANDB = True
NUM_GPUS = 1 if torch.cuda.is_available() else 0
CHECKPOINT_DIR = os.path.abspath("./log")

# ディレクトリ関係
COMMON_CONFIG_PATH = os.path.abspath("./config/config.ini")
CALLBACK_PATH = os.path.abspath("./callback")
RAY_TEMP_DIR = "/tmp"
MODEL_SAVE_FREQ = int(os.environ.get("model_save_freq"))

# 環境関連
BASE_ALIVE_TIME = 35

# モデル関連
HIDDEN_DIM = 128
HIDDEN_DEPTH = 8
TRAINING_ITER = 500
LR = 1e-3

# self-play関連
WIN_RATE_THRESHOLD = float(os.environ.get("win_rate_threshold")) #debug
ALGO = "IMPALA"
FRAMEWORK = "torch"
NUM_ENV_RUNNERS = int(os.environ.get("num_env_runners"))
STOP_TIMESTEPS = int(os.environ.get("stop_timesteps"))
STOP_ITERS = int(os.environ.get("stop_iters"))



# 時間情報を取得
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo("Asia/Tokyo"))
time_code = now.strftime("%Y%m%d_%H:%M:%S")
print(time_code)

# wandbにlogin
import configparser

config_ini = configparser.ConfigParser()
config_ini.read(COMMON_CONFIG_PATH, encoding='utf-8')
api_key = config_ini['WANDB']['api_key']

import wandb
wandb.login(key=api_key)

ge = gym.make('CartPole-v1')
action_space = ge.action_space
observation_space = Box(
    high=np.concatenate([ge.observation_space.high, [1.0], [1.0]]),
    low=np.concatenate([ge.observation_space.low, [0.0], [0.0]]),
    shape=(6,),
    dtype=np.float32
)

def get_action_and_observation_space():
    ge = gym.make('CartPole-v1')
    action_space = ge.action_space
    observation_space = Box(
        high=np.concatenate([ge.observation_space.high, [np.float32(1.0)], [np.float32(1.0)]]),
        low=np.concatenate([ge.observation_space.low, [np.float32(-1.0)], [np.float32(-1.0)]]),
        shape=(6,),
        dtype=np.float32
    )

    return action_space, observation_space


class PvpCartpoleEnv(MultiAgentEnv):
    """Custom CartPole PvP環境"""
    def __init__(self, config):
        """環境の初期化"""
        super().__init__()

        # agentを定義        
        self._agent_ids = {"stabilizer", "disturber"}

        # それぞれのプレイヤーのaction_spaceとobservation_spaceを定義
        single_action_space, single_observation_space = get_action_and_observation_space()
        
        self.action_space = Dict(
            {"stabilizer": single_action_space, "disturber": single_action_space}
        )

        self.observation_space = Dict(
            {"stabilizer": single_observation_space, "disturber": single_observation_space}
        )

        # gym環境を立ち上げ
        self.gym_env = gym.make('CartPole-v1')
        

    def reset(self, *, seed=None, options=None):
        """環境をリセット"""

        self._step = 0
        self.state, info = self.gym_env.reset(seed=seed)
        self._action_player = random.choice(list(self._agent_ids))

        obs = {self._action_player:self.observe(self._action_player)}

        return obs, {}         

    def step(self, action_dict):

        """環境のステップを進める"""
        # stabilizerが勝った時にバグってる？
        # print(f"step! _step is {self._step}")
        action = action_dict.get(self._action_player)
        self.state, _, terminated, truncated, _ = self.gym_env.step(action)
        
        # ゲームオーバーの判定
        self._step += 1
        game_over = terminated or truncated or self._step >= BASE_ALIVE_TIME * 10000000000

        ### 結果の処理 ###
        # ゲームが終わっている場合
        if game_over:
            if self._step >= BASE_ALIVE_TIME:
                # stabilizerの勝利
                win_agent = "stabilizer"
                lose_agent = "disturber"
                s_reward = 1.0
                d_reward = -1.0
            else:
                # disturberの勝利
                win_agent = "disturber"
                lose_agent = "stabilizer"
                s_reward = -1.0
                d_reward = 1.0
            
            # 終了フラグを設定
            self.terminations = {
                self._action_player :True,
                "__all__"   : True
            }
            # print("game over!!!")

        # ゲームが終わってない場合
        else:
            if self._step >= BASE_ALIVE_TIME:
                # stabilizerの勝利
                win_agent = "stabilizer"
                lose_agent = "disturber"
                s_reward = 0.00
                d_reward = -0.00
            else:
                # disturberの勝利
                win_agent = "disturber"
                lose_agent = "stabilizer"
                s_reward = -0.00
                d_reward = 0.00

            self.terminations = {
                "stabilizer":False,
                "disturber" :False,
                "__all__"   :False
            }

        ### 戻り値の準備 ###
        reward = {
            "stabilizer":s_reward,
            "disturber" :d_reward
        }
        self.truncateds = {
            self._action_player :False,
            "__all__": False
        }
        self._action_player = "stabilizer" if self._action_player == "disturber" else "disturber"

        # 次のプレイヤーへのobserve
        obs = {self._action_player: self.observe(self._action_player)}

        return obs, reward, self.terminations, self.truncateds, {}

    def observe(self, agent):
        """指定したエージェントの観測を返す"""
        s = self.add_role_to_state(self.state, agent)
        # s = np.zeros(6).astype(np.float32)
        # assert len(s) == 6, f'Assertion failed: s={s}'
        return s

    def add_role_to_state(self, state, agent_name):
        """観測に役割とターン情報を追加"""

        role = 0.0 if agent_name == "stabilizer" else 1.0
        next_my_turn = 0.0 #使わない

        _state = deepcopy(state)
        _state = np.concatenate([_state, [role, next_my_turn]])
        _state = _state.astype(np.float32)

        return _state

    def render(self):
        """環境のレンダリング"""
        self.gym_env.render()

    def close(self):
        self.gym_env.close()

    # def observation_space(self, agent):
    #     """指定したエージェントの観測空間を返す"""
    #     return self.observation_spaces[agent]

    # def action_space(self, agent):
    #     """指定したエージェントの行動空間を返す"""
    #     return Discrete(2)

from ray.rllib.models.torch.torch_action_dist import TorchCategorical

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

class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # print(f'num_outputs = {num_outputs}')
        # assert num_outputs == 2, f'Assertion failed: num_outputs must be 2 but num_outputs={num_outputs}'
        self.num_outputs = num_outputs
        self._num_objects = obs_space.shape[0]
        self._num_actions = num_outputs

        self.hidden_depth = HIDDEN_DEPTH

        layers = [nn.Linear(self._num_objects, HIDDEN_DIM), nn.ReLU()]
        for i in range(HIDDEN_DEPTH):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        # value function
        self.vf_head = nn.Linear(HIDDEN_DIM, 1)

        # action logits（ソフトマックスは適用しない）
        self.ac_head = nn.Linear(HIDDEN_DIM, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        # assert not isinstance(x, collections.OrderedDict) , f'input is orderdict {x}'
        x = self.layers(x)
        logits = self.ac_head(x)
        self.value_out = self.vf_head(x)
        return logits, []

    def value_function(self):
        return torch.reshape(self.value_out, (-1,))
    # # 新規追加部分: get_weightsメソッドの実装
    # def get_weights(self):
    #     return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

    # # 新規追加部分: set_weightsメソッドの実装
    # def set_weights(self, weights):
    #     state_dict = {k: torch.tensor(v) for k, v in weights.items()}
    #     self.load_state_dict(state_dict)
    



env_name = "pvp_cartpole"
ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)


def env_creator(args):
    env = CustomEnvironment()
    return env


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


# customのselfplay callback
from collections import defaultdict

import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
import shutil
import os
from functools import partial

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

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self, policy_mapper,win_rate_threshold):
        super().__init__()
        self._matching_stats = defaultdict(int)
        self.my_callback_dir = CALLBACK_PATH
        self.model_save_freq = MODEL_SAVE_FREQ
        self.policy_mapper = policy_mapper
        self.win_rate_threshold = win_rate_threshold

    def save_model_checkpoint(self, algorithm):
        policy = algorithm.get_policy("main")
        model = policy.model
        model_state_dict = model.state_dict()

        os.makedirs(self.my_callback_dir, exist_ok=True)
        save_path = os.path.join(self.my_callback_dir, f"main_v{self.policy_mapper.current_opponent}.pt")

        torch.save(model_state_dict, save_path)

    def on_train_result(self, *, algorithm, result, **kwargs):
        
        # 勝率計算
        win_rate_threshold = WIN_RATE_THRESHOLD
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
        



policy_mapper = PolicyMapper()


def main():
    print("do main")

    import ray
    from ray import tune
    from ray.tune import register_env
    import functools

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            log_to_driver=False, 
            _temp_dir=RAY_TEMP_DIR,
            # local_mode=True #debug
        )

    # 利用可能なリソースを取得
    available_resources = ray.cluster_resources()
    num_cpus = int(available_resources.get("CPU", 1))
    num_gpus = int(available_resources.get("GPU", 0))

    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
    from ray.air.constants import TRAINING_ITERATION
    from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule

    # カスタム環境の登録
    
    env_name = "pvp_cartpole"
    register_env(env_name, lambda config: PvpCartpoleEnv(config))

    policy_mapper = PolicyMapper()

    config = (
        get_trainable_cls(ALGO)
        .get_default_config()
        .environment("pvp_cartpole")
        .framework("torch")
        .callbacks(partial(SelfPlayCallback, policy_mapper=policy_mapper, win_rate_threshold=WIN_RATE_THRESHOLD))
        .training(
            lr=LR,
            grad_clip=20.0,
            model={
                "custom_model": "my_torch_model"
            },
            learner_queue_size=10000,
            train_batch_size=50000,
            num_sgd_iter=2,
        )
        .rollouts(
            num_rollout_workers=max(1, (num_cpus)//4),  # num_workersをrolloutsで設定
            num_envs_per_worker=1,  # 各ワーカーでの環境の数を指定
        )
        .env_runners(
            # num_env_runners=max(1, num_cpus // 4),  # 動的に設定: CPU数に応じた環境ランナー
            num_env_runners=max(1, num_cpus-4),  # 動的に設定: CPU数に応じた環境ランナー
            # num_env_runners=3,
            num_envs_per_env_runner=4
            # num_envs_per_env_runner=1
        )
        .learners(
            num_learners=num_gpus,  # 利用可能なGPUに基づいて学習ワーカーを設定
            num_gpus_per_learner=1,  # GPUがある場合にGPUを使用
        )
        .resources(
            num_cpus_for_main_process=4  # メインプロセスに使用するCPUの数
        )
        .multi_agent(
            policies={
                # "main": (CustomPolicy, observation_space, action_space, {"model": {"custom_model": "my_torch_model"}}),
                "main": (None, observation_space, action_space,  {"model": {"custom_model": "my_torch_model"}}),
                "random": (RandomPolicy , observation_space, action_space, {}),
            },
            policy_mapping_fn=policy_mapper,
            policies_to_train=["main"],
        )
    )

    stop = {
        NUM_ENV_STEPS_SAMPLED_LIFETIME: STOP_TIMESTEPS,
        TRAINING_ITERATION: STOP_ITERS
    }

    tune.run(
        ALGO,
        config=config.to_dict(),
        storage_path=CHECKPOINT_DIR,
        callbacks=[
            WandbLoggerCallback(project=PROJ_NAME), #debug
            CSVLoggerCallback()
        ],
        # stop=stop,
    )


if __name__ == "__main__":
    main()
