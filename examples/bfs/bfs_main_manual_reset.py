import argparse
import sys
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom


def train(device_env, cpu_env, step_env, gamma=0.9, max_depth=4, crossover_level=3):
    # Create root index tensor
    root_index = torch.LongTensor([0])
    # Get a list of the minimal set of actions for the environment
    min_actions = device_env.cart.minimal_actions()
    min_actions_size = len(min_actions)

    num_envs = min_actions_size ** max_depth

    device_actions = torch.arange(num_envs, device=device_env.device) % min_actions_size
    device_actions = device_env.action_set[device_actions.long()]
    cpu_actions = device_actions.to(cpu_env.device)

    counter = 0

    while counter < 48000:
        cpu_env.set_size(1)

        # Set device environment root state before calling step function
        cpu_env.states[0] = step_env.states[0]
        cpu_env.ram[0] = step_env.ram[0]
        cpu_env.frame_states[0] = step_env.frame_states[0]

        # Zero out all buffers before calling any environment functions
        cpu_env.rewards.zero_()
        cpu_env.observations1.zero_()
        cpu_env.observations2.zero_()
        cpu_env.done.zero_()

        # Make sure all actions in the backend are completed
        # Be careful making calls to pytorch functions between cule synchronization calls
        if device_env.is_cuda:
            device_env.sync_other_stream()

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)

        total_start.record()

        # Create a default depth_env pointing to the CPU backend
        depth_env = cpu_env
        depth_actions = cpu_actions

        # Perform BFS tree expansion from root state
        for depth in range(max_depth - 1):
            depth_start = torch.cuda.Event(enable_timing=True)
            depth_end = torch.cuda.Event(enable_timing=True)

            depth_start.record()

            # By level 3 there should be enough states to warrant moving to the GPU.
            # We do this by copying all of the relevant state information between the
            # backend GPU and CPU instances.
            if depth == crossover_level:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)

                device_env.set_size(cpu_env.size())
                depth_env = device_env
                depth_actions = device_actions

                copy_start.record()
                device_env.states.copy_(cpu_env.states)
                device_env.ram.copy_(cpu_env.ram)
                device_env.rewards.copy_(cpu_env.rewards)
                device_env.done.copy_(cpu_env.done)
                device_env.frame_states.copy_(cpu_env.frame_states)
                copy_end.record()

                torch.cuda.synchronize()
                device_env.update_frame_states()

                depth_copytime = copy_start.elapsed_time(copy_end)
                print('Depth copy time: {:4.4f} (ms)'.format(depth_copytime))

            # Compute the number of environments at the current depth
            num_envs = min_actions_size ** (depth + 1)
            depth_env.expand(num_envs)

            # Loop over the number of frameskips
            for frame in range(depth_env.frameskip):
                # Execute backend call to the C++ step function with environment data
                super(AtariEnv, depth_env).step(depth_env.fire_reset and depth_env.is_training, \
                                                False, depth_actions.data_ptr(), 0, depth_env.done.data_ptr(), 0)
                # Update the reward, done, and lives flags
                depth_env.get_data(depth_env.episodic_life, gamma**depth, depth_env.done.data_ptr(), depth_env.rewards.data_ptr(), depth_env.lives.data_ptr())

                # To properly compute the output observations we need the last frame AND the second to last frame.
                # On the second to last step we need to update the frame buffers
                if args.debug or (depth == (args.max_depth - 1) and frame == (depth_env.frameskip - 2)):
                    depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels, depth_env.observations2.data_ptr())

                if args.debug:
                    fig = plt.figure()
                    plt.imshow(np.squeeze(np.hstack(depth_env.observations2[:min(num_envs, 4)].cpu())), animated=False, cmap='gray')
                    plt.title('Level: {}, Number of environments: {}'.format(depth + 1, num_envs))
                    plt.show()

            depth_end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            depth_runtime = depth_start.elapsed_time(depth_end)
            # print('Level {} with {} environments: {:4.4f} (ms)'.format(depth + 1, num_envs, depth_runtime))

        # On the last step we call generate_frames again to get the last frames
        depth_env.generate_frames(depth_env.rescale, True, depth_env.num_channels, depth_env.observations1.data_ptr())

        # Make sure all actions in the backend are completed
        if depth_env.is_cuda:
            depth_env.sync_this_stream()
            torch.cuda.current_stream().synchronize()

        # Form observations using max of last 2 frame_buffers
        torch.max(depth_env.observations1, depth_env.observations2, out=depth_env.observations1)

        total_end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        total_runtime = total_start.elapsed_time(total_end)
        # print('Total expansion time: {:4.4f} (ms)'.format(total_runtime))

        # Compute max over rewards to find the best action
        # assert(depth_env.rewards.size(0) % (depth_env.action_space.n ** (max_depth - 1)) == 0)
        best_value = depth_env.rewards.max()
        best_action = depth_env.rewards.argmax() // depth_env.action_space.n ** (max_depth - 1)

        best_action = step_env.action_set[best_action.long()]
        observation, rewards, done, info = step_env.step(best_action.unsqueeze(-1))

        counter += 1

        # Reset the step env here if we want to play another game
        if done[0]:
            random_index = np.random.randint(0, step_env.cached_states.size(0))
            step_env.states[0] = step_env.cached_states[random_index]
            step_env.ram[0] = step_env.cached_ram[random_index]
            step_env.frame_states[0] = step_env.cached_frame_states[random_index]


def bfs_main(args):
    device = torch.device('cuda', args.gpu) if args.use_cuda_env else torch.device('cpu')

    cart = AtariRom(args.env_name)
    num_actions = len(cart.minimal_actions())
    num_envs = num_actions ** (args.max_depth - 1)

    # Create an environment for processing BFS steps on the GPU
    device_env = AtariEnv(args.env_name, num_envs, color_mode='gray', repeat_prob=0.0,
                          device=device, rescale=True, episodic_life=args.episodic_life, frameskip=4)
    device_env.train()
    super(AtariEnv, device_env).reset(0)

    # Create an environment for processing BFS steps on the CPU
    cpu_env = AtariEnv(args.env_name, num_envs, color_mode='gray', repeat_prob=0.0,
                       device='cpu', rescale=True, episodic_life=args.episodic_life, frameskip=4)
    cpu_env.train()
    super(AtariEnv, cpu_env).reset(0)

    # Create an environment for stepping
    step_env = AtariEnv(args.env_name, 1, color_mode='gray', repeat_prob=0.0, device='cpu',
                        rescale=True, episodic_life=args.episodic_life, frameskip=4, max_noop_steps=args.max_noop_steps)
    step_env.train()
    step_env.reset(initial_steps=args.ale_start_steps, verbose=args.verbose)

    action = torch.Tensor([0], dtype=torch.uint8)
    num_random_steps = 40
    for i in range(args.max_noop_steps):
        for j in range(num_random_steps):
            step_env.step(action)

    train(device_env, cpu_env, step_env, args.gamma, args.max_depth, args.crossover_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('--ale-start-steps', type=int, default=400, help='Number of steps used to initialize ALEs (default: 400)')
    parser.add_argument('--crossover-level', type=int, default=3, help='Level to start crossover from CPU to GPU execution (default: 3)')
    parser.add_argument('--clip-rewards', action='store_true', default=False, help='Clip rewards to {-1, 0, +1}')
    parser.add_argument('--debug', action='store_true', default=False, help='show ALE frames')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='Atari game name')
    parser.add_argument('--episodic-life', action='store_true', default=False, help='use end of life as end of episode')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: None)')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='random seed (default: time())')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--use-cuda-env', action='store_true', default=False, help='use CUDA for ALE updates')
    parser.add_argument('--use-openai', action='store_true', default=False, help='Use OpenAI Gym environment')
    parser.add_argument('--max-depth', type=int, default=4, help='depth of action space for BFS traversal (default: 4)')
    parser.add_argument('--max-noop-steps', type=int, default=200, help='depth of action space for BFS traversal (default: 200)')
    args = parser.parse_args(sys.argv[1:])

    bfs_main(args)
