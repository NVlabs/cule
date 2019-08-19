from datetime import datetime
import os
import pytz
import time
import torch

import torch.nn.functional as F
import numpy as np

total_time = 0
last_save = 0

def gen_data(x):
    return [f(x).item() for f in [torch.mean, torch.median, torch.min, torch.max, torch.std]]

def format_time(f):
    return datetime.fromtimestamp(f, tz=pytz.utc).strftime('%H:%M:%S.%f s')

def evaluate(args, frames, exec_time, model, env, csv_writer, csv_file):
    lengths, rewards = test(args, model, env)
    lengths, rewards = lengths.float(), rewards.float()

    lmean, lmedian, lmin, lmax, lstd = gen_data(lengths)
    rmean, rmedian, rmin, rmax, rstd = gen_data(rewards)
    length_data = '(length) min/max/mean/median: {lmin:4.1f}/{lmax:4.1f}/{lmean:4.1f}/{lmedian:4.1f}'.format(lmin=lmin, lmax=lmax, lmean=lmean, lmedian=lmedian)
    reward_data = '(reward) min/max/mean/median: {rmin:4.1f}/{rmax:4.1f}/{rmean:4.1f}/{rmedian:4.1f}'.format(rmin=rmin, rmax=rmax, rmean=rmean, rmedian=rmedian)
    print('[training time: {}] {}'.format(format_time(exec_time), ' --- '.join([length_data, reward_data])))

    if csv_writer and csv_file:
        csv_writer.writerow([frames,exec_time,
                             rmean,rmedian,rmin,rmax,rstd,
                             lmean,lmedian,lmin,lmax,lstd])
        csv_file.flush()

    return lengths, rewards

def test(args, policy_net, env):
    width, height = 84, 84
    num_ales = args.evaluation_episodes

    if args.use_openai_test_env:
        observation = torch.from_numpy(env.reset()).squeeze(1)
    else:
        observation = env.reset(initial_steps=3).squeeze(-1)

    lengths = torch.zeros(num_ales, dtype=torch.int32)
    rewards = torch.zeros(num_ales, dtype=torch.int32)
    all_done = torch.zeros(num_ales, dtype=torch.bool)
    not_done = torch.ones(num_ales, dtype=torch.int32)

    fire_reset = torch.zeros(num_ales, dtype=torch.bool)
    actions = torch.ones(num_ales, dtype=torch.uint8)

    maybe_npy = lambda a: a.numpy() if args.use_openai_test_env else a

    info = env.step(maybe_npy(actions))[-1]
    if args.use_openai_test_env:
        lives = torch.IntTensor([d['ale.lives'] for d in info])
    else:
        lives = info['ale.lives'].clone()

    states = torch.zeros((num_ales, args.num_stack, width, height), device='cuda', dtype=torch.float32)
    states[:, -1] = observation.to(device='cuda', dtype=torch.float32)

    policy_net.eval()

    while not all_done.all():
        logit = policy_net(states)[1]

        actions = F.softmax(logit, dim=1).multinomial(1).cpu()
        actions[fire_reset] = 1

        observation, reward, done, info = env.step(maybe_npy(actions))

        if args.use_openai_test_env:
            # convert back to pytorch tensors
            observation = torch.from_numpy(observation)
            reward = torch.from_numpy(reward.astype(np.int32))
            done = torch.from_numpy(done.astype(np.bool))
            new_lives = torch.IntTensor([d['ale.lives'] for d in info])
        else:
            new_lives = info['ale.lives'].clone()

        done = done.bool()
        fire_reset = new_lives < lives
        lives.copy_(new_lives)

        observation = observation.to(device='cuda', dtype=torch.float32)

        states[:, :-1].copy_(states[:, 1:].clone())
        states *= (1.0 - done.to(device='cuda', dtype=torch.float32)).view(-1, *[1] * (observation.dim() - 1))
        states[:, -1].copy_(observation.view(-1, *states.size()[-2:]))

        # update episodic reward counters
        lengths += not_done
        rewards += reward.cpu() * not_done.cpu()

        all_done |= done.cpu()
        all_done |= (lengths >= args.max_episode_length)
        not_done = (all_done == False).int()

    policy_net.train()

    return lengths, rewards

def callback(args, model, frames, iter_time, rewards, lengths,
             value_loss, policy_loss, entropy, csv_writer, csv_file):
    global last_save, total_time

    if not hasattr(args, 'num_steps_per_update'):
        args.num_steps_per_update = args.num_steps

    total_time += iter_time
    fps = (args.world_size * args.num_steps_per_update * args.num_ales) / iter_time
    lmean, lmedian, lmin, lmax, lstd = gen_data(lengths)
    rmean, rmedian, rmin, rmax, rstd = gen_data(rewards)

    if frames >= last_save:
        last_save += args.save_interval

        # torch.save(model.state_dict(), args.model_name)

        if csv_writer and csv_file:
            csv_writer.writerow([frames, fps, total_time,
                                 rmean, rmedian, rmin, rmax, rstd,
                                 lmean, lmedian, lmin, lmax, lstd,
                                 entropy, value_loss, policy_loss])
            csv_file.flush()

    str_template = '{fps:8.2f}f/s, ' \
                   'min/max/mean/median reward: {rmin:5.1f}/{rmax:5.1f}/{rmean:5.1f}/{rmedian:5.1f}, ' \
                   'entropy/value/policy: {entropy:6.4f}/{value:6.4f}/{policy: 6.4f}'

    return str_template.format(fps=fps, rmin=rmin, rmax=rmax, rmean=rmean, rmedian=rmedian,
                               entropy=entropy, value=value_loss, policy=policy_loss)
