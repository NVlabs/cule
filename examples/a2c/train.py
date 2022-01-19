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
    train_csv_file, train_csv_writer, eval_csv_file, eval_csv_writer, summary_writer = log_initialize(args, train_device)
    train_env, test_env, observation = env_initialize(args, env_device)

    model = ActorCritic(args.num_stack, train_env.action_space, normalize=args.normalize, name=args.env_name)
    model, optimizer = model_initialize(args, model, train_device)

    num_frames_per_iter = args.num_ales * args.num_steps
    total_steps = math.ceil(args.t_max / (args.world_size * num_frames_per_iter))

    shape = (args.num_steps + 1, args.num_ales, args.num_stack, *train_env.observation_space.shape[-2:])
    states = torch.zeros(shape, device=train_device, dtype=torch.float32)
    states[0, :, -1] = observation.to(device=train_device, dtype=torch.float32)

    shape = (args.num_steps + 1, args.num_ales)
    values  = torch.zeros(shape, device=train_device, dtype=torch.float32)
    returns = torch.zeros(shape, device=train_device, dtype=torch.float32)

    shape = (args.num_steps, args.num_ales)
    rewards = torch.zeros(shape, device=train_device, dtype=torch.float32)
    masks = torch.zeros(shape, device=train_device, dtype=torch.float32)
    actions = torch.zeros(shape, device=train_device, dtype=torch.long)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    episode_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)

    if args.use_gae:
        gae = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)

    maybe_npy = lambda a: a.numpy() if args.use_openai else a

    torch.cuda.synchronize()

    iterator = range(total_steps)
    if args.rank == 0:
        iterator = tqdm(iterator)
        total_time = 0
        evaluation_offset = 0

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

    for update in iterator:

        T = args.world_size * update * num_frames_per_iter
        if (args.rank == 0) and (T >= evaluation_offset):
            evaluation_offset += args.evaluation_interval
            eval_lengths, eval_rewards = test(args, model, test_env)

            lmean, lmedian, lmin, lmax, lstd = gen_data(eval_lengths)
            rmean, rmedian, rmin, rmax, rstd = gen_data(eval_rewards)
            length_data = '(length) min/max/mean/median: {lmin:4.1f}/{lmax:4.1f}/{lmean:4.1f}/{lmedian:4.1f}'.format(lmin=lmin, lmax=lmax, lmean=lmean, lmedian=lmedian)
            reward_data = '(reward) min/max/mean/median: {rmin:4.1f}/{rmax:4.1f}/{rmean:4.1f}/{rmedian:4.1f}'.format(rmin=rmin, rmax=rmax, rmean=rmean, rmedian=rmedian)
            print('[training time: {}] {}'.format(format_time(total_time), ' --- '.join([length_data, reward_data])))

            if eval_csv_writer and eval_csv_file:
                eval_csv_writer.writerow([T, total_time, rmean, rmedian, rmin, rmax, rstd, lmean, lmedian, lmin, lmax, lstd])
                eval_csv_file.flush()

            if args.plot:
                summary_writer.add_scalar('eval/rewards_mean', rmean, T, walltime=total_time)
                summary_writer.add_scalar('eval/lengths_mean', lmean, T, walltime=total_time)

        start_time = time.time()

        with torch.no_grad():

            for step in range(args.num_steps):
                value, logit = model(states[step])

                # store values
                values[step] = value.squeeze(-1)

                # convert actions to numpy and perform next step
                probs_action = F.softmax(logit, dim=1).multinomial(1).to(env_device)
                observation, reward, done, info = train_env.step(maybe_npy(probs_action))

                if args.use_openai:
                    # convert back to pytorch tensors
                    observation = torch.from_numpy(observation)
                    reward = torch.from_numpy(reward)
                    done = torch.from_numpy(done.astype(np.uint8))
                else:
                    observation = observation.squeeze(-1).unsqueeze(1)

                # move back to training memory
                observation = observation.to(device=train_device)
                reward = reward.to(device=train_device, dtype=torch.float32)
                done = done.to(device=train_device, dtype=torch.bool)
                probs_action = probs_action.to(device=train_device, dtype=torch.long)

                not_done = 1.0 - done.float()

                # update rewards and actions
                actions[step].copy_(probs_action.view(-1))
                masks[step].copy_(not_done)
                rewards[step].copy_(reward.sign())

                # update next observations
                states[step + 1, :, :-1].copy_(states[step, :, 1:].clone())
                states[step + 1] *= not_done.view(-1, *[1] * (observation.dim() - 1))
                states[step + 1, :, -1].copy_(observation.view(-1, *states.size()[-2:]))

                # update episodic reward counters
                episode_rewards += reward
                final_rewards[done] = episode_rewards[done]
                episode_rewards *= not_done

                episode_lengths += not_done
                final_lengths[done] = episode_lengths[done]
                episode_lengths *= not_done

            returns[-1] = values[-1] = model(states[-1])[0].data.squeeze(-1)

            if args.use_gae:
                gae.zero_()
                for step in reversed(range(args.num_steps)):
                    delta = rewards[step] + (args.gamma * values[step + 1] * masks[step]) - values[step]
                    gae = delta + (args.gamma * args.tau * masks[step] * gae)
                    returns[step] = gae + values[step]
            else:
                for step in reversed(range(args.num_steps)):
                    returns[step] = rewards[step] + (args.gamma * returns[step + 1] * masks[step])

        value, logit = model(states[:-1].view(-1, *states.size()[-3:]))

        log_probs = F.log_softmax(logit, dim=1)
        probs = F.softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, actions.view(-1).unsqueeze(-1))
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        advantages = returns[:-1].view(-1).unsqueeze(-1) - value

        value_loss = advantages.pow(2).mean()
        policy_loss = -(advantages.clone().detach() * action_log_probs).mean()

        loss = value_loss * args.value_loss_coef + policy_loss - dist_entropy * args.entropy_coef
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        states[0].copy_(states[-1])

        torch.cuda.synchronize()

        if args.rank == 0:
            iter_time = time.time() - start_time
            total_time += iter_time

            if args.plot:
                summary_writer.add_scalar('train/rewards_mean', final_rewards.mean().item(), T, walltime=total_time)
                summary_writer.add_scalar('train/lengths_mean', final_lengths.mean().item(), T, walltime=total_time)
                summary_writer.add_scalar('train/value_loss', value_loss, T, walltime=total_time)
                summary_writer.add_scalar('train/policy_loss', policy_loss, T, walltime=total_time)
                summary_writer.add_scalar('train/entropy', dist_entropy, T, walltime=total_time)

                # print(states[0,:32].size())
                # images = np.squeeze(np.hstack(states[0,:32].cpu()))
                # summary_writer.add_image('train/frames', images, T, walltime=total_time)

            progress_data = callback(args, model, T, iter_time, final_rewards, final_lengths,
                                     value_loss.item(), policy_loss.item(), dist_entropy.item(),
                                     train_csv_writer, train_csv_file)
            iterator.set_postfix_str(progress_data)

    if args.plot and (args.rank == 0):
        summary_writer.close()

    if args.use_openai:
        train_env.close()
    if args.use_openai_test_env:
        test_env.close()
