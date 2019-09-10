import torch

import torch.nn.functional as F
import numpy as np

def test(args, policy_net, env):
    device = next(policy_net.parameters()).device

    width, height = 84, 84
    num_ales = args.evaluation_episodes

    if args.use_openai_test_env:
        observation = torch.from_numpy(env.reset()).squeeze(1)
    else:
        observation = env.reset(initial_steps=50).squeeze(-1)

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

    states = torch.zeros((num_ales, args.num_stack, width, height), device=device, dtype=torch.float32)
    states[:, -1] = observation.to(device=device, dtype=torch.float32)

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

        observation = observation.to(device=device, dtype=torch.float32)

        states[:, :-1].copy_(states[:, 1:].clone())
        states *= (1.0 - done.to(device=device, dtype=torch.float32)).view(-1, *[1] * (observation.dim() - 1))
        states[:, -1].copy_(observation.view(-1, *states.size()[-2:]))

        # update episodic reward counters
        lengths += not_done
        rewards += reward.cpu() * not_done.cpu()

        all_done |= done.cpu()
        all_done |= (lengths >= args.max_episode_length)
        not_done = (all_done == False).int()

    policy_net.train()

    return lengths, rewards
