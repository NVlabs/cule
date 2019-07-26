import math
import sys
import torch

import numpy as np

class ReplayMemory():
    def __init__(self, args, capacity, device, num_ales=None):
        self.num_ales = num_ales if num_ales else args.num_ales
        self.device = device
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.priority_replay = args.priority_replay

        self.full = False
        self.index = 0
        self.epoch = 0
        self.steps_per_ale = int(math.ceil(float(capacity) / self.num_ales))
        self.capacity = self.steps_per_ale * self.num_ales

        self.actions = torch.zeros(self.num_ales, self.steps_per_ale, device=self.device, dtype=torch.long)

        if self.priority_replay:
            self.priority = torch.ones(self.capacity, device=self.device, dtype=torch.float32) * float(np.finfo(np.float32).eps)
            self.priority_view = self.priority.view(self.num_ales, self.steps_per_ale)
            self.rank = torch.zeros(self.capacity, device=self.device, dtype=torch.long)

        self.section_offset = torch.zeros(0, device=self.device, dtype=torch.int32)
        self.section_size = torch.zeros(0, device=self.device, dtype=torch.int32)
        self.gammas = self.discount ** torch.FloatTensor(range(self.n)).to(self.device).unsqueeze(0)
        self.frame_offsets = torch.IntTensor(range(-(self.history - 1), 1)).to(device=self.device).unsqueeze(0)
        self.weights = torch.ones(args.batch_size, device=self.device, dtype=torch.float32)

        width, height = 84, 84
        imagesize = width * height
        num_steps = self.steps_per_ale + 2 * (self.history - 1)
        stepsize  = num_steps * imagesize

        self.observations = torch.zeros((self.num_ales, num_steps, width, height), device=self.device, dtype=torch.uint8)
        self.states_view = torch.zeros(0, device=self.device, dtype=torch.uint8)
        self.states_view.set_(self.observations.storage(),
                              storage_offset=0,
                              size=torch.Size([self.num_ales, num_steps, self.history, width, height]),
                              stride=(stepsize, imagesize, imagesize, width, 1))

        self.frame_number = torch.zeros(self.num_ales, num_steps, device=self.device, dtype=torch.int32)
        self.frame_number[:, (self.history - 1) + (self.steps_per_ale - 1)] = -1
        self.frame_view = torch.zeros(0, device=self.device, dtype=torch.int32)
        self.frame_view.set_(self.frame_number.storage(),
                             storage_offset=0,
                             size=torch.Size([self.num_ales, num_steps, self.history]),
                             stride=(num_steps, 1, 1))

        self.rewards = torch.zeros(self.num_ales, num_steps, device=self.device, dtype=torch.float32)
        self.reward_view = torch.zeros(0, device=self.device, dtype=torch.float32)
        self.reward_view.set_(self.rewards.storage(),
                              storage_offset=0,
                              size=torch.Size([self.num_ales, num_steps, self.n]),
                              stride=(num_steps, 1, 1))

    def update_sections(self, batch_size):
        capacity = self.capacity if self.full else self.index * self.num_ales

        if self.section_size.size(0) != capacity:
            # initialize rank-based priority segment boundaries
            pdf = torch.FloatTensor(1.0 / np.arange(1, capacity + 1)).to(device=self.device) ** self.priority_exponent
            self.p_i_sum = pdf.sum(0)
            pdf = pdf / self.p_i_sum
            cdf = pdf.cumsum(0)

            haystack = cdf.cpu().numpy()
            needles = np.linspace(0, 1, batch_size, endpoint=False)[::-1]
            self.section_offset = np.trim_zeros(np.searchsorted(haystack, needles))
            self.section_offset = torch.from_numpy(self.section_offset).to(device=self.device)
            self.section_offset = torch.cat((self.section_offset, torch.LongTensor([0]).to(device=self.device)))
            self.section_size = self.section_offset[:-1] - self.section_offset[1:]

            mask = self.section_size != 0
            self.section_offset = self.section_offset[:-1][mask]
            self.section_offset = torch.cat(((self.section_offset, torch.LongTensor([0]).to(device=self.device))))
            self.section_size = self.section_size[mask]

        return self.section_size.size(0)

    def reset(self, observations):
        self.observations[:, self.history - 1] = observations.mul(255.0).to(device=self.device, dtype=torch.uint8)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, observations, actions, rewards, terminals):
        if actions is None:
            actions = torch.zeros(self.num_ales, device=self.device, dtype=torch.long)
        if rewards is None:
            rewards = torch.zeros(self.num_ales, device=self.device, dtype=torch.float32)

        curr_offset = self.index + self.history
        prev_offset = curr_offset - 1 + ((self.index == 0) * self.steps_per_ale)

        nonterminal = (terminals == 0).float()
        terminal_mask = terminals == 1

        self.actions[:, self.index] = actions
        self.rewards[:, self.index] = rewards.float() * nonterminal
        self.rewards[terminal_mask, prev_offset - (self.history - 1)] = rewards[terminal_mask].float()
        self.observations[:, curr_offset] = observations.mul(255.0).to(device=self.device, dtype=torch.uint8)
        self.frame_number[:, curr_offset] = nonterminal.int() * (self.frame_number[:, prev_offset] + 1)
        # self.frame_number[:, curr_offset] += terminals.int() * np.random.randint(sys.maxsize / 1024)

        if self.priority_replay:
            self.priority_view[:, self.index] = self.priority.max() + float(np.finfo(np.float32).eps)
            self.rank = self.priority.sort(descending=True)[1]

        if (self.epoch > 0) and (self.index == 0):
            self.observations[:, self.history - 1] = self.observations[:, self.steps_per_ale + self.history - 1]
            self.frame_number[:, self.history - 1] = self.frame_number[:, self.steps_per_ale + self.history - 1]

        self.index = (self.index + 1) % self.steps_per_ale  # Update index
        self.full |= self.index == 0
        self.epoch += int(self.index == 0)

    def sample(self, batch_size=0, indices=None):
        capacity = self.capacity if self.full else self.index * self.num_ales

        if indices is None:
            indices = torch.randint(capacity, (batch_size,), device=self.device, dtype=torch.long)

        batch_size = indices.size(0)

        if self.priority_replay:
            batch_size = self.update_sections(batch_size)

            indices = self.section_offset[:-1] + (indices[:batch_size] % self.section_size)
            p_i = (indices.float() + 1.0) ** -self.priority_exponent
            P = p_i / self.p_i_sum
            indices = self.rank[indices]

            weights = (capacity * P) ** -self.priority_weight  # Compute importance-sampling weights w
            weights /= weights.max()   # Normalize by max importance-sampling weight from batch
        else:
            weights = self.weights

        ale_indices = indices % self.num_ales
        step_indices = indices / self.num_ales

        # Create un-discretised state and nth next state
        base_frame_numbers = self.frame_number[ale_indices, step_indices + self.history - 1].unsqueeze(-1).expand(-1, self.history)
        expected_frame_numbers = base_frame_numbers + self.frame_offsets.expand(batch_size, -1)

        actual_frame_numbers = self.frame_view[ale_indices, step_indices]
        states_mask = (actual_frame_numbers == expected_frame_numbers).float().unsqueeze(-1).unsqueeze(-1)
        states = self.states_view[ale_indices, step_indices].float().div_(255.0) * states_mask

        next_actual_frame_numbers = self.frame_view[ale_indices, step_indices + self.n]
        next_states_mask = (next_actual_frame_numbers == (expected_frame_numbers + self.n)).float().unsqueeze(-1).unsqueeze(-1)
        next_states = self.states_view[ale_indices, step_indices + self.n].float().div_(255.0) * next_states_mask

        # Discrete action to be used as index
        actions = self.actions[ale_indices, step_indices]

        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = self.reward_view[ale_indices, step_indices]
        returns = torch.sum(self.gammas * rewards * next_states_mask[:, -self.n - 1:-1, 0, 0], 1)

        # Check validity of the last state
        nonterminals = next_actual_frame_numbers[:, -1] == (expected_frame_numbers[:, -1] + self.n)

        return indices, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, indices, td_error):
        if self.priority_replay:
            self.priority[indices] = td_error.abs() ** self.priority_exponent
            self.rank = self.priority.sort(descending=True)[1]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration

        ale_index = int(self.current_idx % self.num_ales)
        step_index = int(self.current_idx / self.num_ales)

        # Create un-discretised state
        base_frame_numbers = self.frame_number[ale_index, step_index + self.history - 1].expand(self.history)
        expected_frame_numbers = base_frame_numbers + self.frame_offsets.squeeze(0)

        actual_frame_numbers = self.frame_view[ale_index, step_index]
        states_mask = (actual_frame_numbers == expected_frame_numbers).float().unsqueeze(-1).unsqueeze(-1)
        states = self.states_view[ale_index, step_index].float().div_(255.0) * states_mask

        self.current_idx += 1
        return states

