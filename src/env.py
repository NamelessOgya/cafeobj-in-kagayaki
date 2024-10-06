import gymnasium as gym
import functools
import random
from copy import copy, deepcopy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

import collections
import os

import yaml

with open("training_config.yaml", 'r') as file:
    training_config = yaml.safe_load(file)    

BASE_ALIVE_TIME = os.environ.get("base_alive_time")



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

def env_creator(args):
    env = CustomEnvironment()
    return env


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
                s_reward = 0.01
                d_reward = -0.01
            else:
                # disturberの勝利
                win_agent = "disturber"
                lose_agent = "stabilizer"
                s_reward = -0.01
                d_reward = 0.01

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


