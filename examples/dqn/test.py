import numpy as np
import os
import torch
import time

from memory import ReplayMemory
from torchcule.atari import Env as AtariEnv
from utils.openai.envs import create_vectorize_atari_env
from helper import format_time

def test(args, T, dqn, val_mem, env, device):

    if args.use_openai:
        observation = torch.from_numpy(env.reset()).squeeze(1)
    else:
        observation = env.reset(initial_steps=10).clone().squeeze(-1)

    num_ales = args.evaluation_episodes

    # Test performance over several episodes
    state = torch.zeros((num_ales, args.history_length, 84, 84), dtype=torch.float32, device=device)
    state[:, -1] = observation.float().div_(255.0)

    # These variables are used to compute average rewards for all processes.
    lengths = torch.zeros(num_ales, dtype=torch.float32)
    rewards = torch.zeros(num_ales, dtype=torch.float32)
    all_done = torch.zeros(num_ales, dtype=torch.uint8)
    not_done = torch.ones(num_ales, dtype=torch.float32)

    fire_reset = torch.zeros(num_ales, dtype=torch.uint8)
    actions = torch.ones(num_ales, dtype=torch.uint8)

    maybe_npy = lambda a: a.numpy() if args.use_openai else a

    info = env.step(maybe_npy(actions))[-1]
    if args.use_openai:
        lives = torch.IntTensor([d['ale.lives'] for d in info])
    else:
        lives = info['ale.lives'].clone()

    while not all_done.all():
        actions = dqn.act(state).cpu()  # Choose an action ε-greedily

        actions[fire_reset] = 1
        observation, reward, done, info = env.step(maybe_npy(actions))  # Step

        if args.use_openai:
            # convert back to pytorch tensors
            observation = torch.from_numpy(observation)
            reward = torch.from_numpy(reward)
            done = torch.from_numpy(done.astype(np.uint8))
            observation = observation.squeeze(1)
        else:
            observation = observation.clone().squeeze(-1)

        if args.use_openai:
            new_lives = torch.IntTensor([d['ale.lives'] for d in info])
        else:
            new_lives = info['ale.lives'].clone()
        fire_reset = new_lives < lives
        lives.copy_(new_lives)

        state[:, :-1].copy_(state[:, 1:].clone())
        state *= not_done.to(device=device).view(-1, 1, 1, 1)
        state[:, -1].copy_(observation.to(device=device, dtype=torch.float32).div(255.0))

        # update episodic reward counters
        lengths += not_done
        rewards += reward.float() * not_done

        all_done |= done
        all_done |= (lengths >= args.max_episode_length)
        not_done = (all_done == False).float()

    # Test Q-values over validation memory
    Qs = []

    for state in val_mem:  # Iterate over valid states
        Qs.append(dqn.evaluate_q(state))

    avg_Q = sum(Qs) / len(Qs)

    # Return average reward and Q-value
    return rewards, lengths, avg_Q

def initialize_validation(args, device):
    val_mem = ReplayMemory(args, args.evaluation_size, device=device, num_ales=1)

    val_env = AtariEnv(args.env_name, 1, color_mode='gray',
                       device='cpu', rescale=True,
                       clip_rewards=args.reward_clip,
                       episodic_life=True, repeat_prob=0.0)
    val_env.train()

    observation = val_env.reset(initial_steps=100, verbose=False).clone().to(device=device, dtype=torch.float32).squeeze(-1).div_(255.0)
    val_mem.reset(observation)

    for _ in range(val_mem.capacity):
        observation, _, done, info = val_env.step(val_env.sample_random_actions())
        observation = observation.clone().to(device=device, dtype=torch.float32).squeeze(-1).div_(255.0)
        done = done.to(device=device)
        val_mem.append(observation, None, None, done)

    return val_mem

