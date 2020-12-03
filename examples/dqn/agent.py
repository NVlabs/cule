import os
import torch

import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from model import DQN

class Agent():
  def __init__(self, args, action_space):
      self.action_space = action_space
      self.n = args.multi_step
      self.discount = args.discount
      self.target_update = args.target_update
      self.categorical = args.categorical
      self.noisy_linear = args.noisy_linear
      self.double_q = args.double_q
      self.max_grad_norm = args.max_grad_norm
      self.device = torch.device('cuda', args.gpu)
      self.num_param_updates = 0

      if args.categorical:
          self.atoms = args.atoms
          self.v_min = args.v_min
          self.v_max = args.v_max
          self.support = torch.linspace(self.v_min, args.v_max, self.atoms).to(device=self.device)  # Support (range) of z
          self.delta_z = (args.v_max - self.v_min) / (self.atoms - 1)

      self.online_net = DQN(args, self.action_space.n).to(device=self.device)

      if args.model and os.path.isfile(args.model):
          # Always load tensors onto CPU by default, will shift to GPU if necessary
          self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
      self.online_net.train()

      self.target_net = DQN(args, self.action_space.n).to(device=self.device)
      self.update_target_net()
      self.target_net.eval()
      for param in self.target_net.parameters():
          param.requires_grad = False

      self.optimizer = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps, amsgrad=True)

      if args.distributed:
          self.online_net = DDP(self.online_net)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
      if isinstance(self.online_net, DQN):
          self.online_net.reset_noise()
      else:
          self.online_net.module.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      probs = self.online_net(state.to(self.device))
      if self.categorical:
          probs = self.support.expand_as(probs) * probs
      actions = probs.sum(-1).argmax(-1).to(state.device)
      return actions

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    actions = self.act(state)

    mask = torch.rand(state.size(0), device=state.device, dtype=torch.float32) < epsilon
    masked = mask.sum().item()
    if masked > 0:
        actions[mask] = torch.randint(0, self.action_space.n, (masked,), device=state.device, dtype=torch.long)

    return actions

  def learn(self, states, actions, returns, next_states, nonterminals, weights):

    tactions = actions.unsqueeze(-1).unsqueeze(-1)
    if self.categorical:
        tactions = tactions.expand(-1, -1, self.atoms)

    # Calculate current state probabilities (online network noise already sampled)
    nvtx.range_push('agent:online (state) probs')
    ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    ps_a = ps.gather(1, tactions)  # log p(s_t, a_t; θonline)
    nvtx.range_pop()

    with torch.no_grad():
      if isinstance(self.target_net, DQN):
          self.target_net.reset_noise()
      else:
          self.target_net.module.reset_noise()  # Sample new target net noise

      nvtx.range_push('agent:target (next state) probs')
      tns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      nvtx.range_pop()

      if self.double_q:
        # Calculate nth next state probabilities
        nvtx.range_push('agent:online (next state) probs')
        pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
        nvtx.range_pop()
      else:
        pns = tns

      if self.categorical:
        pns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

      # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      argmax_indices_ns = pns.sum(-1).argmax(-1).unsqueeze(-1).unsqueeze(-1)
      if self.categorical:
          argmax_indices_ns = argmax_indices_ns.expand(-1, -1, self.atoms)
      pns_a = tns.gather(1, argmax_indices_ns)  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      if self.categorical:
        # Compute Tz (Bellman operator T applied to z)
        # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = returns.unsqueeze(-1) + nonterminals.float().unsqueeze(-1) * (self.discount ** self.n) * self.support.unsqueeze(0)
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.v_min) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        batch_size = states.size(0)
        m = states.new_zeros(batch_size, self.atoms)
        offset = torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size).unsqueeze(1).expand(batch_size, self.atoms).to(actions)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a.squeeze(1) * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a.squeeze(1) * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
      else:
        Tz = returns + nonterminals.float() * (self.discount ** self.n) * pns_a.squeeze(-1).squeeze(-1)

    if self.categorical:
        loss = -torch.sum(m * ps_a.squeeze(1), 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        weights = weights.unsqueeze(-1)
    else:
        loss = F.mse_loss(ps_a.squeeze(-1).squeeze(-1), Tz, reduction='none')

    nvtx.range_push('agent:loss + step')
    self.optimizer.zero_grad()
    weighted_loss = (weights * loss).mean()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
    self.optimizer.step()
    nvtx.range_pop()

    return loss.detach()

  def update_target_net(self):
      self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path):
      torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
      with torch.no_grad():
          q = self.online_net(state.unsqueeze(0).to(self.device))
          if self.categorical:
              q *= self.support
          return q.sum(-1).max(-1)[0].item()

  def train(self):
      self.online_net.train()

  def eval(self):
      self.online_net.eval()

  def __str__(self):
      return self.online_net.__str__()
