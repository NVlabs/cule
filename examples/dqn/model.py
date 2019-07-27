import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from utils.openai.vec_normalize import RunningMeanStd

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()

        self.categorical = args.categorical
        self.dueling = args.dueling
        self.atoms = args.atoms if args.categorical else 1
        self.action_space = action_space

        Linear = NoisyLinear if args.noisy_linear else nn.Linear

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=args.history_length, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),		
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),		
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU()
                )

        conv_out_size = self._get_conv_out((args.history_length, 84, 84))

        # TODO: Add std_init argument to noisy linear constructors
        self.fc_a = nn.Sequential(
                Linear(in_features=conv_out_size, out_features=args.hidden_size),
                nn.ReLU(),
                Linear(in_features=args.hidden_size, out_features=action_space * self.atoms),
                )

        if args.dueling:
            self.fc_v = nn.Sequential(
                    Linear(in_features=conv_out_size, out_features=args.hidden_size),
                    nn.ReLU(),
                    Linear(in_features=args.hidden_size, out_features=self.atoms),
                    )

        self.apply(weights_init)

        self.ob_rms = RunningMeanStd(shape=(84, 84)) if args.normalize else None

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, log=False):
        with torch.no_grad():
            if self.ob_rms:
                if self.training:
                    self.ob_rms.update(x)
                mean = self.ob_rms.mean.to(dtype=torch.float32, device=x.device)
                std = torch.sqrt(self.ob_rms.var.to(dtype=torch.float32, device=x.device) + float(np.finfo(np.float32).eps))
                x = (x - mean) / std

        conv_out = self.conv(x).view(x.size(0), -1)
        a = self.fc_a(conv_out).view(-1, self.action_space, self.atoms)

        if self.dueling:
            v = self.fc_v(conv_out).view(-1, 1, self.atoms)
            q = v + a - a.mean(1, keepdim=True) # Combine streams
        else:
            q = a

        if self.categorical:
            if log:  # Use log softmax for numerical stability
                q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
            else:
                q = F.softmax(q, dim=2)  # Probabilities with action over second dimension

        return q

    def reset_noise(self):
        for m in self.fc_a.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

        if self.dueling:
            for m in self.fc_v.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
