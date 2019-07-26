import gym
from gym.spaces import Box

from .atari_wrappers import make_atari, wrap_deepmind
from .subproc_vec_env import SubprocVecEnv

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def create_atari_env(env_id, seed=0, rank=0, episode_life=False, clip_rewards=False, deepmind=True, max_frames=18000):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)
        if deepmind:
            env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards)
            env = WrapPyTorch(env)
        return env
    return _thunk

def create_vectorize_atari_env(env_id, seed, num_envs, episode_life=False, clip_rewards=False, deepmind=True, max_frames=18000):
    return SubprocVecEnv([create_atari_env(env_id,
                                           seed=seed,
                                           rank=proc_id,
                                           episode_life=episode_life,
                                           clip_rewards=clip_rewards,
                                           deepmind=deepmind,
                                           max_frames=max_frames) for proc_id in range(num_envs)])

