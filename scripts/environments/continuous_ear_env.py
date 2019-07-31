import os
import math
import sys
from collections import defaultdict

import gym
from gym import error, spaces, utils

from gym.utils import seeding

import numpy as np
import random


class EarEnv(gym.Env):

    def __init__(self):
        super(EarEnv, self).__init__()

        train_y = np.load("output/train_anno.npy")
        train_x = np.load("output/train_latent.npy")
        
        _, output_dim = train_y.shape
        num_train, input_dim = train_x.shape

        # pos[3], norm[3]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(output_dim,), dtype=np.float32)
        #self.action_space = spaces.Box(low=np.array([1.2, 50.0]), high=np.array([2.0, 75.0]), dtype=np.float32)

        # latent representation, sample index
        self.observation_space = spaces.Box(low=0, high=num_train, shape=(1,), dtype=np.int)

  
        self.viewer = None
        self.steps_beyond_done = None
        self.seed()

        self.observation = self.reset()

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #action_ = [2*action[0]/100.0, action[1]]
        sample_idx = self._get_obs()
        state, gt_action = train_x[sample_idx], train_y[sample_idx]
        
        res_obs = self._get_outcome(action,gt_action)
        gain = self._get_gain_factor(obs, res_obs)

        # Calculate reward
        reward = self._get_reward(gain)

        done = True
        return obs, reward, done, {'is_success':self._is_success(reward,1.0),
                                   'res_obs':res_obs,
                                   'gain_factor':gain}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._get_obs()

    # standin for output weight
    def _get_outcome(self, obs, action):
        if obs > 4500.0:
            g = self.srm.predict([action[0],action[1]])
        else:
            g = self.srm_m.predict([action[0],action[1]])
        g = g * random.uniform(0.9, 1.1)
        return obs*(1+g*0.01)

    def _get_gain_factor(self, obs, res_obs):
        return 1+(res_obs-obs)/obs

    def _get_obs(self):
        return self.observation_space.sample()

    def _get_reward(self,g):
        """Reward is given for an injection that comes close to the desired gain without overshooting."""
        if g > self.MAX_GAIN:
            #return 0.0 # Reward is given for coming close to the desired gain without overshooting.
            return self.MAX_GAIN/g
        else:
            return g/self.MAX_GAIN

    def _is_success(self, achieved, desired):
        return abs(achieved-desired) < 0.1
