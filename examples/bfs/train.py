import math
import time
import torch

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from utils.initializers import args_initialize, env_initialize, log_initialize, model_initialize

from helper import callback, format_time, gen_data
from model import ActorCritic
from test import test

def worker(gpu, ngpus_per_node, args):
    env_device, train_device = args_initialize(gpu, ngpus_per_node, args)
    train_env, test_env, observation = env_initialize(args, env_device, bfs_depth=4)

    model = ActorCritic(args.num_stack, train_env.action_space, normalize=args.normalize, name=args.env_name)
    # model, optimizer = model_initialize(args, model, train_device)
    #
    # num_frames_per_iter = args.num_ales * args.num_steps
    # total_steps = math.ceil(args.t_max / (args.world_size * num_frames_per_iter))
    #
    # shape = (args.num_steps + 1, args.num_ales, args.num_stack, *train_env.observation_space.shape[-2:])
    # states = torch.zeros(shape, device=train_device, dtype=torch.float32)
    # states[0, :, -1] = observation.to(device=train_device, dtype=torch.float32)
    #
    #
    # if args.use_openai:
    #     train_env.close()
    # if args.use_openai_test_env:
    #     test_env.close()
