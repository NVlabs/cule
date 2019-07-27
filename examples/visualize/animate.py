import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from torchcule.atari import Env, Rom

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('--color', type=str, default='rgb', help='Color mode (rgb or gray)')
    parser.add_argument('--clip-rewards', action='store_true', help='Clip rewards to be {-1, 0, +1}')
    parser.add_argument('--debug', action='store_true', help='Single step through frames for debugging')
    parser.add_argument('--env-name', type=str, help='Atari Game')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--initial-steps', type=int, default=1000, help='Number of steps used to initialize the environment')
    parser.add_argument('--num-envs', type=int, default=5, help='Number of atari environments')
    parser.add_argument('--rescale', action='store_true', help='Resize output frames to 84x84 using bilinear interpolation')
    parser.add_argument('--training', action='store_true', help='Set environment to training mode')
    parser.add_argument('--use-cuda', action='store_true', help='Execute ALEs on GPU')
    args = parser.parse_args()

    cmap   = None if args.color == 'rgb' else 'gray'
    device = torch.device('cuda:{}'.format(args.gpu) if args.use_cuda else 'cpu')
    debug  = args.debug

    env = Env(args.env_name, args.num_envs, args.color, device=device, rescale=args.rescale,
              episodic_life=True, clip_rewards=args.clip_rewards, repeat_prob=0.0)
    print(env.cart)

    if args.training:
        env.train()
    observations = env.reset(initial_steps=args.initial_steps, verbose=True).cpu().numpy()

    fig = plt.figure()
    img = plt.imshow(np.squeeze(np.hstack(observations)), animated=True, cmap=cmap)
    ax = fig.add_subplot(111)

    frame = 0

    if debug:
        ax.set_title("frame: {}, rewards: {}, done: {}".format(frame, [], []))
    else:
        fig.suptitle(frame)

    def updatefig(*args):
        global ax, debug, env, frame, img

        if debug:
            input("Press Enter to continue...")

        actions = env.sample_random_actions()
        observations, reward, done, info = env.step(actions)
        observations = observations.cpu().numpy()
        reward = reward.cpu().numpy()
        done = done.cpu().numpy()
        img.set_array(np.squeeze(np.hstack(observations)))

        if debug:
            ax.title.set_text("{}) rewards: {}, done: {}".format(frame, reward, done))
        else:
            fig.suptitle(frame)

        frame += 1

        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

