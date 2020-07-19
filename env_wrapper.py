import gym
#import cv2 as cv
import numpy as np
from rl_manipulation_gym_envs.src.rl_manipulation_gym_envs.puzzle_env import PuzzleEnv
from gym import wrappers
from utils import standardize_image

class Environment:
    def __init__(self, env_name, rand_seed):
        self.env = gym.make(env_name)
        self.env.seed(rand_seed)

    def env_info(self):
        s_dim = self.env.observation_space.shape
        a_dim = self.env.action_space.shape[0]

        return [None, s_dim[0], s_dim[1], s_dim[2]], [None, a_dim]

    def reset(self):
        return self._process_state(self.env.reset())

    def step(self, a):
        s_next, r, terminal, _ = self.env.step(action=self._process_action(a))
        return self._process_state(s_next), self._process_reward(r), terminal

    def _process_action(self, a):
        # TODO proprocessing of action
        a *= 2.
        return a

    def _process_state(self, s):
        # TODO preprocessing of state
        #s = np.reshape(s, (96, 96))
        #s = np.expand_dims(cv.cvtColor(s, cv.COLOR_RGB2GRAY), axis=2)
        #s = s / 255.
        #s = standardize_image(s)
        #s = cv.resize(s, (80, 80))
        s[2] = s[2] / 8
        return s

    def _process_reward(self, r):
        # TODO preprocessing of reward
        r = r / 16.2736044
        #r = np.clip(r, min_r, max_r)
        return r