import numpy as np
import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

from utils.openai.vec_normalize import RunningMeanStd

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, action_space, normalize=False, name=None):
        super(ActorCritic, self).__init__()

        self._name = name

        self.conv1 = nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out((num_inputs, 84, 84))
        self.linear1 = nn.Linear(in_features=conv_out_size, out_features=512)

        self.critic_linear = nn.Linear(in_features=512, out_features=1)
        self.actor_linear = nn.Linear(in_features=512, out_features=action_space.n)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        self.ob_rms = RunningMeanStd(shape=(84, 84)) if normalize else None

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        with torch.no_grad():
            if self.ob_rms:
                if self.training:
                    self.ob_rms.update(x)
                mean = self.ob_rms.mean.to(dtype=torch.float32, device=x.device)
                std = torch.sqrt(self.ob_rms.var.to(dtype=torch.float32, device=x.device) + float(np.finfo(np.float32).eps))
                x = (x - mean) / std

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)

    def name(self):
        return self._name

    def save(self):
        if self.name():
            name = '{}.pth'.format(self.name())
            torch.save(self.state_dict(), name)

    def load(self, name=None):
        self.load_state_dict(torch.load(name if name else self.name()))

