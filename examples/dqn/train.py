import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

import csv
import json
import numpy as np
import math
import pytz
import random
import socket
import time
import torch
import torch.cuda.nvtx as nvtx

from datetime import datetime
from tqdm import tqdm
from torchcule.atari import Env as AtariEnv

from agent import Agent
from memory import ReplayMemory
from test import initialize_validation, test

from utils.openai.envs import create_vectorize_atari_env
from utils.runtime import cuda_device_str

class data_prefetcher():
    def __init__(self, batch_size, device, mem):
        self.batch_size = batch_size
        self.device = device
        self.mem = mem
        self.stream = torch.cuda.Stream()

    def preload(self):
        with torch.cuda.stream(self.stream):
            self.idxs, self.states, self.actions, self.returns, self.next_states, self.nonterminals, self.weights = self.mem.sample(self.batch_size)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        idxs = self.idxs.to(device=self.device)
        states = self.states.to(device=self.device)
        actions = self.actions.to(device=self.device)
        returns = self.returns.to(device=self.device)
        next_states = self.next_states.to(device=self.device)
        nonterminals = self.nonterminals.to(device=self.device)
        weights = self.weights.to(device=self.device)

        self.preload()
        return idxs, states, actions, returns, next_states, nonterminals, weights

def train(args):
	args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
	args.distributed = (args.world_size > 1) or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count() if args.num_gpus_per_node == -1 else args.num_gpus_per_node
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		torch.multiprocessing.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		worker(args.local_rank, ngpus_per_node, args)

def vec_stats(vec):
    return [func(vec).item() for func in [torch.mean, torch.median, torch.std, torch.min, torch.max]]

def format_time(f):
    return datetime.fromtimestamp(f, tz=pytz.utc).strftime('%H:%M:%S.%f s')

def worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        args.seed += args.gpu
        torch.cuda.set_device(args.gpu)

        args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu

        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8632',
                                             world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    args.use_cuda_env = args.use_cuda_env and torch.cuda.is_available()
    args.no_cuda_train = not torch.cuda.is_available()
    args.verbose = args.verbose and (args.rank == 0)

    env_device = torch.device('cuda', args.gpu) if args.use_cuda_env else torch.device('cpu')
    train_device = torch.device('cuda', args.gpu) if (args.no_cuda_train == False) else torch.device('cpu')

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if args.use_cuda_env or (args.no_cuda_train == False):
        torch.cuda.manual_seed(random.randint(1, 10000))

    if train_device.type == 'cuda':
        print('Train:\n' + cuda_device_str(train_device.index), flush=True)

    if args.use_openai:
        test_env = create_vectorize_atari_env(args.env_name, args.seed, args.evaluation_episodes,
                                              episode_life=False, clip_rewards=False)
        test_env.reset()
    else:
        test_env = AtariEnv(args.env_name, args.evaluation_episodes, color_mode='gray',
                            device='cpu', rescale=True, clip_rewards=False,
                            episodic_life=False, repeat_prob=0.0, frameskip=4)

    # Agent
    dqn = Agent(args, test_env.action_space)

    # Construct validation memory
    if args.rank == 0:
        print('Initializing evaluation memory with {} entries...'.format(args.evaluation_size), end='', flush=True)
        start_time = time.time()

    val_mem = initialize_validation(args, train_device)

    if args.rank == 0:
        print('complete ({})'.format(format_time(time.time() - start_time)), flush=True)

    if args.evaluate:
        if args.rank == 0:
            eval_start_time = time.time()
            dqn.eval()  # Set DQN (online network) to evaluation mode
            rewards, lengths, avg_Q = test(args, 0, dqn, val_mem, test_env, train_device)
            dqn.train()  # Set DQN (online network) back to training mode
            eval_total_time = time.time() - eval_start_time

            rmean, rmedian, rstd, rmin, rmax = vec_stats(rewards)
            lmean, lmedian, lstd, lmin, lmax = vec_stats(lengths)

            print('reward: {:4.2f}, {:4.0f}, {:4.0f}, {:4.4f} | '
                  'length: {:4.2f}, {:4.0f}, {:4.0f}, {:4.4f} | '
                  'Avg. Q: {:4.4f} | {}'
                  .format(rmean, rmin, rmax, rstd, lmean, lmin, lmax,
                          lstd, avg_Q, format_time(eval_total_time)),
                  flush=True)
    else:
        if args.rank == 0:
            print('Entering main training loop', flush=True)

            if args.output_filename:
                csv_file = open(args.output_filename, 'w', newline='')
                csv_file.write(json.dumps(vars(args)))
                csv_file.write('\n')
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow(['frames', 'total_time',
                                     'rmean', 'rmedian', 'rstd', 'rmin', 'rmax',
                                     'lmean', 'lmedian', 'lstd', 'lmin', 'lmax'])
            else:
                csv_writer, csv_file = None, None

            if args.plot:
                from tensorboardX import SummaryWriter
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                log_dir = os.path.join(args.log_dir, current_time + '_' + socket.gethostname())
                writer = SummaryWriter(log_dir=log_dir)
                for k, v in vars(args).items():
                    writer.add_text(k, str(v))

            # Environment
            print('Initializing environments...', end='', flush=True)
            start_time = time.time()

        if args.use_openai:
            train_env = create_vectorize_atari_env(args.env_name, args.seed, args.num_ales,
                                                   episode_life=True, clip_rewards=args.reward_clip,
                                                   max_frames=args.max_episode_length)
            observation = torch.from_numpy(train_env.reset()).squeeze(1)
        else:
            train_env = AtariEnv(args.env_name, args.num_ales, color_mode='gray',
                                 device=env_device, rescale=True,
                                 clip_rewards=args.reward_clip,
                                 episodic_life=True, repeat_prob=0.0)
            train_env.train()
            observation = train_env.reset(initial_steps=args.ale_start_steps, verbose=args.verbose).clone().squeeze(-1)

        if args.rank == 0:
            print('complete ({})'.format(format_time(time.time() - start_time)), flush=True)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
        episode_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
        final_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
        final_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
        has_completed = torch.zeros(args.num_ales, device=train_device, dtype=torch.uint8)

        mem = ReplayMemory(args, args.memory_capacity, train_device)
        mem.reset(observation)
        priority_weight_increase = (1 - args.priority_weight) / (args.t_max - args.learn_start)

        state = torch.zeros((args.num_ales, args.history_length, 84, 84), device=mem.device, dtype=torch.float32)
        state[:, -1] = observation.to(device=mem.device, dtype=torch.float32).div(255.0)

        num_frames_per_iter = args.num_ales
        total_steps = math.ceil(args.t_max / (args.world_size * num_frames_per_iter))
        epsilons = np.linspace(args.epsilon_start, args.epsilon_final, math.ceil(args.epsilon_frames / num_frames_per_iter))
        epsilon_offset = math.ceil(args.learn_start / num_frames_per_iter)

        prefetcher = data_prefetcher(args.batch_size, train_device, mem)

        avg_loss = 'N/A'
        eval_offset = 0
        target_update_offset = 0

        total_time = 0

        # main loop
        iterator = range(total_steps)
        if args.rank == 0:
            iterator = tqdm(iterator)

        env_stream = torch.cuda.Stream()
        train_stream = torch.cuda.Stream()

        for update in iterator:

            T = args.world_size * update * num_frames_per_iter
            epsilon = epsilons[min(update - epsilon_offset, len(epsilons) - 1)] if T >= args.learn_start else epsilons[0]
            start_time = time.time()

            if update % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            dqn.eval()
            nvtx.range_push('train:select action')
            if args.noisy_linear:
                action = dqn.act(state)  # Choose an action greedily (with noisy weights)
            else:
                action = dqn.act_e_greedy(state, epsilon=epsilon)
            nvtx.range_pop()
            dqn.train()

            if args.use_openai:
                action = action.cpu().numpy()

            torch.cuda.synchronize()

            with torch.cuda.stream(env_stream):
                nvtx.range_push('train:env step')
                if args.use_openai:
                    observation, reward, done, info = train_env.step(action)  # Step
                    # convert back to pytorch tensors
                    observation = torch.from_numpy(observation).squeeze(1)
                    reward = torch.from_numpy(reward.astype(np.float32))
                    done = torch.from_numpy(done.astype(np.uint8))
                    action = torch.from_numpy(action)
                else:
                    observation, reward, done, info = train_env.step(action, asyn=True)  # Step
                    observation = observation.clone().squeeze(-1)
                nvtx.range_pop()

                observation = observation.to(device=train_device)
                reward = reward.to(device=train_device)
                done = done.to(device=train_device)
                action = action.to(device=train_device)

                observation = observation.float().div_(255.0)

                state[:, :-1].copy_(state[:, 1:].clone())
                state *= (1 - done).view(-1, 1, 1, 1).float()
                state[:, -1].copy_(observation)

                # update episodic reward counters
                not_done = (1 - done).float()
                has_completed |= (done == 1)

                episode_rewards += reward.float()
                final_rewards[done] = episode_rewards[done]
                episode_rewards *= not_done

                episode_lengths += not_done
                final_lengths[done] = episode_lengths[done]
                episode_lengths *= not_done

            # Train and test
            if T >= args.learn_start:
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                prefetcher.preload()

                avg_loss = 0.0
                num_minibatches = min(int(args.num_ales / args.replay_frequency), 8)
                for _ in range(num_minibatches):
                    # Sample transitions
                    nvtx.range_push('train:sample states')
                    idxs, states, actions, returns, next_states, nonterminals, weights = prefetcher.next()
                    nvtx.range_pop()

                    nvtx.range_push('train:network update')
                    loss = dqn.learn(states, actions, returns, next_states, nonterminals, weights)
                    nvtx.range_pop()

                    nvtx.range_push('train:update priorities')
                    mem.update_priorities(idxs, loss)  # Update priorities of sampled transitions
                    nvtx.range_pop()

                    avg_loss += loss.mean().item()
                avg_loss /= num_minibatches

                # Update target network
                if T >= target_update_offset:
                    dqn.update_target_net()
                    target_update_offset += args.target_update

            torch.cuda.current_stream().wait_stream(env_stream)
            torch.cuda.current_stream().wait_stream(train_stream)

            nvtx.range_push('train:append memory')
            mem.append(observation, action, reward, done)  # Append transition to memory
            nvtx.range_pop()

            total_time += time.time() - start_time

            if args.rank == 0:
                if args.plot and ((update % args.replay_frequency) == 0):
                    writer.add_scalar('train/epsilon', epsilon, T)
                    writer.add_scalar('train/rewards', final_rewards.mean(), T)
                    writer.add_scalar('train/lengths', final_lengths.mean(), T)

                if T >= eval_offset:
                    eval_start_time = time.time()
                    dqn.eval()  # Set DQN (online network) to evaluation mode
                    rewards, lengths, avg_Q = test(args, T, dqn, val_mem, test_env, train_device)
                    dqn.train()  # Set DQN (online network) back to training mode
                    eval_total_time = time.time() - eval_start_time
                    eval_offset += args.evaluation_interval

                    rmean, rmedian, rstd, rmin, rmax = vec_stats(rewards)
                    lmean, lmedian, lstd, lmin, lmax = vec_stats(lengths)

                    print('reward: {:4.2f}, {:4.0f}, {:4.0f}, {:4.4f} | '
                          'length: {:4.2f}, {:4.0f}, {:4.0f}, {:4.4f} | '
                          'Avg. Q: {:4.4f} | {}'
                          .format(rmean, rmin, rmax, rstd, lmean, lmin, lmax,
                                  lstd, avg_Q, format_time(eval_total_time)),
                          flush=True)

                    if args.output_filename and csv_writer and csv_file:
                        csv_writer.writerow([T, total_time,
                                             rmean, rmedian, rstd, rmin, rmax,
                                             lmean, lmedian, lstd, lmin, lmax])
                        csv_file.flush()

                    if args.plot:
                        writer.add_scalar('eval/rewards', rmean, T)
                        writer.add_scalar('eval/lengths', lmean, T)
                        writer.add_scalar('eval/avg_Q', avg_Q, T)

                loss_str = '{:4.4f}'.format(avg_loss) if isinstance(avg_loss, float) else avg_loss
                progress_data = 'T = {:,} epsilon = {:4.2f} avg reward = {:4.2f} loss: {}' \
                                .format(T, epsilon, final_rewards.mean().item(), loss_str)
                iterator.set_postfix_str(progress_data)

    if args.plot and (args.rank == 0):
        writer.close()

    if args.use_openai:
        train_env.close()
        test_env.close()
