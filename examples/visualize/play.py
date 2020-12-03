import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import torch
import torch.nn.functional as F

from torchcule.atari import Env as AtariEnv

_path = os.path.abspath(os.path.pardir)
sys.path = [os.path.join(_path, 'a2c')] + sys.path
from model import ActorCritic

def downsample(frame, height=84, width=84):
    frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(frame[:, :, None]).byte()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('game', type=str, help='Atari ROM filename')
    parser.add_argument('--num-stack', type=int, default=4, help='number of images in a stack (default: 4)')
    args = parser.parse_args()
    num_stack = args.num_stack

    env = AtariEnv(args.game, num_envs=1)
    env.eval()

    model = ActorCritic(num_stack, env.action_space)
    shape = (args.num_stack, 84, 84)
    states = torch.ByteTensor(*shape).zero_()

    observation = env.reset()[0]
    states[-1] = downsample(observation).squeeze(-1)
    actions = env.minimal_actions()
    N = actions.size(0)

    options = {'noop': 0, 'right': 1, 'left': 2, 'down': 4, 'up': 8, ' ': 16}
    action_keys = [0, 1, 2, 4, 8, 16, 9, 10, 5, 6, 24, 17, 18, 20, 25, 26, 21, 22]
    action_names = ['NOOP', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'FIRE', 'UPRIGHT', \
                    'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', \
                    'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',      \
                    'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
    action_dict = dict(zip(action_keys,action_names))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    img1 = ax1.imshow(observation.numpy(), animated=True)
    ax1.axis('off')
    plt.tight_layout()
    plt.title('Player Observation')

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    inds = range(N)
    img2 = ax2.bar(inds, np.zeros(N))
    plt.xticks(inds, [action_dict[l.item()] for l in actions])
    plt.xticks(rotation=25)
    plt.ylim(ymin=0, ymax=1)
    plt.tight_layout()
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Network Probabilities')

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    img3 = ax3.imshow(np.hstack(states.numpy()), animated=True, cmap='gray')
    ax3.axis('off')
    plt.tight_layout()
    plt.title('Network State')

    imgs = [img1, img3]

    player_a_action = torch.zeros(1).byte()
    player_b_action = torch.zeros(1).byte()

    if args.game == 'pong':
        options['right'], options['left'], options['up'], options['down'] = \
                options['down'], options['up'], options['right'], options['left']

    def on_key_press(event):
        global player_b_action, options
        if event.key in options:
            player_a_action[0] |= options[event.key]

    def on_key_release(event):
        global player_a_action, options
        if event.key in options:
            player_a_action[0] &= ~options[event.key]

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    counter = 0

    def updatefig(*args):
        global env, player_a_action, player_b_action, model, states, counter, num_stack, img2, imgs

        observation = env.step(player_a_action, player_b_action)[0][0]
        imgs[0].set_array(observation.numpy())

        if ((counter % num_stack) == 0) or (counter < num_stack):
            states[:-1] = states[1:].clone()
            states[-1] = downsample(observation).squeeze(-1)
            imgs[1].set_array(np.hstack(states.numpy()))

            with torch.no_grad():
                value, logit = model(states.unsqueeze(0).float())
                probs = F.softmax(logit, dim=1)
                player_b_action = probs.max(1, keepdim=True)[1].data.byte()

            for b, h in zip(img2, probs[0]):
                b.set_height(h)

        counter += 1

        return img2.get_children() + imgs

    ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)

    plt.show()

