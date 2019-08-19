"""CuLE (CUda Learning Environment module)

This module provides access to several RL environments that generate data
on the CPU or GPU.
"""

import atari_py
import math
import numpy as np
import os
import sys
import torch

from gym import spaces

from torchcule.atari.rom import Rom
import torchcule_atari

class Env(torchcule_atari.AtariEnv):
    """
    ALE (Atari Learning Environment)

    This class provides access to ALE environments that may be executed on the CPU
    or GPU.

    Example:
		import argparse
		from torchcule.atari import Env

		parser = argparse.ArgumentParser(description="CuLE")
		parser.add_argument("game", type=str,
							help="Atari game name (breakout)")
		parser.add_argument("--n", type=int, default=20,
							help="Number of atari environments")
		parser.add_argument("--s", type=int, default=200,
							help="Number steps/frames to generate per environment")
		parser.add_argument("--c", type=str, default='rgb',
							help="Color mode (rgb or gray)")
		parser.add_argument("--rescale", action='store_true',
							help="Resize output frames to 84x84 using bilinear interpolation")
		args = parser.parse_args()

		color_mode = args.c
		num_envs = args.n
		num_steps = args.s

		env = Env(args.game, num_envs, color_mode, args.rescale)
		observations = env.reset()

		for _ in np.arange(num_steps):
			actions = env.sample_random_actions()
			observations, reward, done, info = env.step(actions)
    """

    def __init__(self, env_name, num_envs, color_mode='rgb', device='cpu', rescale=False,
                 frameskip=1, repeat_prob=0.25, clip_rewards=False, episodic_life=False,
                 max_noop_steps=30, max_episode_length=10000):
        """Initialize the ALE class with a given environment

        Args:
            env_name (str): The name of the Atari rom
            num_envs (int): The number of environments to run
            color_mode (str): RGB ('rgb') or grayscale ('gray') observations
            use_cuda (bool) : Map ALEs to GPU
            rescale (bool) : Rescale grayscale observations to 84x84
            frameskip (int) : Number of frames to skip during training
            repeat_prob (float) : Probability of repeating previous action
            clip_rewards (bool) : Apply rewards clipping to {-1,1}
            episodic_life (bool) : Set 'done' on end of life
        """

        assert (color_mode == 'rgb') or (color_mode == 'gray')
        if color_mode == 'rgb' and rescale:
            raise ValueError('Rescaling is only valid in grayscale color mode')

        self.cart = Rom(env_name)
        super(Env, self).__init__(self.cart, num_envs, max_noop_steps)

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.rescale = rescale
        self.frameskip = frameskip
        self.repeat_prob = repeat_prob
        self.is_cuda = self.device.type == 'cuda'
        self.is_training = False
        self.clip_rewards = clip_rewards
        self.episodic_life = episodic_life
        self.height = 84 if self.rescale else self.cart.screen_height()
        self.width = 84 if self.rescale else self.cart.screen_width()
        self.num_channels = 3 if color_mode == 'rgb' else 1

        self.action_set = torch.Tensor([int(s) for s in self.cart.minimal_actions()]).to(self.device).byte()

        # check if FIRE is in the action set
        self.fire_reset = torchcule_atari.FIRE in self.action_set

        self.action_space = spaces.Discrete(self.action_set.size(0))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_channels, self.height, self.width), dtype=np.uint8)

        self.observations = torch.zeros((num_envs, self.height, self.width, self.num_channels), device=self.device, dtype=torch.uint8)
        self.done = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.actions = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.last_actions = torch.zeros(num_envs, device=self.device, dtype=torch.uint8)
        self.lives = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.rewards = torch.zeros(num_envs, device=self.device, dtype=torch.int32)

        self.states = torch.zeros((num_envs, self.state_size()), device=self.device, dtype=torch.uint8)
        self.frame_states = torch.zeros((num_envs, self.frame_state_size()), device=self.device, dtype=torch.uint8)
        self.ram = torch.randint(0, 255, (num_envs, self.cart.ram_size()), device=self.device, dtype=torch.uint8)
        self.tia = torch.zeros((num_envs, self.tia_update_size()), device=self.device, dtype=torch.int32)
        self.frame_buffer = torch.zeros((num_envs, 300 * self.cart.screen_width()), device=self.device, dtype=torch.uint8)
        self.cart_offsets = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.rand_states = torch.randint(0, np.iinfo(np.int32).max, (num_envs,), device=self.device, dtype=torch.int32)
        self.cached_states = torch.zeros((max_noop_steps, self.state_size()), device=self.device, dtype=torch.uint8)
        self.cached_ram = torch.randint(0, 255, (max_noop_steps, self.cart.ram_size()), device=self.device, dtype=torch.uint8)
        self.cached_frame_states = torch.zeros((max_noop_steps, self.frame_state_size()), device=self.device, dtype=torch.uint8)
        self.cached_tia = torch.zeros((max_noop_steps, self.tia_update_size()), device=self.device, dtype=torch.int32)
        self.cache_index = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)

        self.set_cuda(self.is_cuda, self.device.index if self.is_cuda else -1)
        self.initialize(self.states.data_ptr(),
                        self.frame_states.data_ptr(),
                        self.ram.data_ptr(),
                        self.tia.data_ptr(),
                        self.frame_buffer.data_ptr(),
                        self.cart_offsets.data_ptr(),
                        self.action_set.data_ptr(),
                        self.rand_states.data_ptr(),
                        self.cached_states.data_ptr(),
                        self.cached_ram.data_ptr(),
                        self.cached_frame_states.data_ptr(),
                        self.cached_tia.data_ptr(),
                        self.cache_index.data_ptr());

    def to(self, device):
        if self.is_cuda:
            torch.cuda.current_stream().synchronize()
            self.sync_this_stream()
            self.sync_other_stream()

        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        self.set_cuda(self.is_cuda, self.device.index if self.is_cuda else -1)

        self.observations = self.observations.to(self.device)
        self.done = self.done.to(self.device)
        self.actions = self.actions.to(self.device)
        self.last_actions = self.last_actions.to(self.device)
        self.lives = self.lives.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.action_set = self.action_set.to(self.device)

        self.states = self.states.to(self.device)
        self.frame_states = self.frame_states.to(self.device)
        self.ram = self.ram.to(self.device)
        self.tia = self.tia.to(self.device)
        self.frame_buffer = self.frame_buffer.to(self.device)
        self.cart_offsets = self.cart_offsets.to(self.device)
        self.rand_states = self.rand_states.to(self.device)
        self.cached_states = self.cached_states.to(self.device)
        self.cached_ram = self.cached_ram.to(self.device)
        self.cached_frame_states = self.cached_frame_states.to(self.device)
        self.cached_tia = self.cached_tia.to(self.device)
        self.cache_index = self.cache_index.to(self.device)

        self.initialize(self.states.data_ptr(),
                        self.frame_states.data_ptr(),
                        self.ram.data_ptr(),
                        self.tia.data_ptr(),
                        self.frame_buffer.data_ptr(),
                        self.cart_offsets.data_ptr(),
                        self.action_set.data_ptr(),
                        self.rand_states.data_ptr(),
                        self.cached_states.data_ptr(),
                        self.cached_ram.data_ptr(),
                        self.cached_frame_states.data_ptr(),
                        self.cached_tia.data_ptr(),
                        self.cache_index.data_ptr());

        if self.is_cuda:
            torch.cuda.current_stream().synchronize()
            self.sync_this_stream()
            self.sync_other_stream()

    def train(self, frameskip=4):
        """Set ALE to training mode"""
        self.frameskip = frameskip
        self.is_training = True

    def eval(self):
        """Set ALE to evaluation mode"""
        self.is_training = False

    def minimal_actions(self):
        """Minimal number of actions for the environment

        Returns:
            list[Action]: minimal set of actions for the environment
        """
        return self.action_set

    def sample_random_actions(self, asyn=False):
        """Generate a random set of actions

        Returns:
            list[Action]: random set of actions generated for the environment
        """
        return torch.randint(self.minimal_actions().size(0), (self.num_envs,), device=self.device, dtype=torch.uint8)

    def screen_shape(self):
        """Get the shape of the observations

        Returns:
            tuple(int,int): Tuple containing height and width of observations
        """
        return (self.height, self.width)

    def reset(self, seeds=None, initial_steps=50, verbose=False, asyn=False):
        """Reset the environments

        Args:
            seeds (list[int]): seeds to use for initialization
            initial_steps (int): number of initial NOOP steps to execute during initialization

        Returns:
            tuple(int,int): Tuple containing height and width of observations
        """
        if seeds is None:
            seeds = torch.randint(np.iinfo(np.int32).max, (self.num_envs,), dtype=torch.int32, device=self.device)

        if self.is_cuda:
            self.sync_other_stream()
            stream = torch.cuda.current_stream()

        super(Env, self).reset(seeds.data_ptr())

        if self.is_training:
            iterator = range(math.ceil(initial_steps / self.frameskip))

            if verbose:
                from tqdm import tqdm
                iterator = tqdm(iterator)

            for _ in iterator:
                actions = self.sample_random_actions()
                self.step(actions, asyn=True)

        if self.is_cuda:
            self.sync_this_stream()
            if not asyn:
                stream.synchronize()

        return self.observations

    def step(self, actions, asyn=False):
        """Take a step in the environment by apply a set of actions

        Args:
            actions (list[Action]): list of actions to apply to each environment

        Returns:
            ByteTensor: observations for each environment
            IntTensor: sum of rewards for frameskip steps in each environment
            ByteTensor: 'done' state for each environment
            list[str]: miscellaneous information (currently unused)
        """
        # sanity checks
        assert actions.size(0) == self.num_envs

        self.rewards.zero_()
        self.observations.zero_()
        self.done.zero_()

        self.actions = self.action_set[actions.long()]

        if self.is_cuda:
            self.sync_other_stream()

        for _ in range(self.frameskip):
            super(Env, self).step(self.fire_reset and self.is_training, self.actions.data_ptr(), self.done.data_ptr())
            self.get_data(self.episodic_life, self.done.data_ptr(), self.rewards.data_ptr(), self.lives.data_ptr())

        self.reset_states()
        self.generate_frames(self.rescale, self.num_channels, self.observations.data_ptr())

        if self.is_cuda:
            self.sync_this_stream()
            if not asyn:
                torch.cuda.current_stream().synchronize()

        if self.clip_rewards:
            self.rewards.sign_()

        info = {'ale.lives': self.lives}

        return self.observations, self.rewards, self.done, info
