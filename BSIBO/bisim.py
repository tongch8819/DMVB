import torch
import torch.nn as nn
import torch.functional as F
import dmc2gym
import argparse
# from gym.spaces import Box
import numpy as np
from collections import defaultdict
import os
import pickle
from torch.optim import Adam

class BisimMetric(nn.Module):
    def __init__(self, obs_shape, num_filters=32):
        """
        obs_shape: (3, 100, 100)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=num_filters, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1),
        )
        out_dim = (((obs_shape[1] - 3) // 2 + 1) - 2) - 2
        self.trunk = nn.Sequential(
            nn.Linear(num_filters * out_dim * out_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, s1, s2):
        o1 = self.conv(s1)
        o2 = self.conv(s2)
        batch_size = o1.shape[0]
        o1 = o1.view(batch_size, -1)
        o2 = o2.view(batch_size, -1)
        x = torch.cat([o1, o2], dim=-1)
        x = self.trunk(x)
        res = x
        return res

class ReplayBuffer:
    def __init__(self, capicity, obs_shape, action_shape, batch_size, device):
        self.capacity = capicity
        self.batch_size = batch_size
        self.device = device
        obs_dtype = np.float32
        self.obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones  = np.empty((self.capacity, 1), dtype=np.float32)

        self.action_to_bin = defaultdict(list)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def get_other_pairs(self, action):
        """
        Continuous case
        """
        bin_idx = self._action_to_bin_idx(action)
        if self.action_to_bin.get(bin_idx, None) is None:
            return None
        idx_pool = self.action_to_bin[bin_idx]
        idxs = np.random.choice(idx_pool, min(len(idx_pool), self.batch_size))
        
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()
        return obses, actions, rewards, next_obses, not_dones

    def _action_to_bin_idx(self, action):
        # TODO: maybe too empirical
        return int((action + 1.) / .1)

    def append(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)  # deep copy
        np.copyto(self.next_obses[self.idx], next_obs)  # deep copy
        np.copyto(self.actions[self.idx], action)  # deep copy
        np.copyto(self.rewards[self.idx], reward)  # deep copy
        np.copyto(self.not_dones[self.idx], not done)  # deep copy

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        # update action_to_bin
        bin_idx = self._action_to_bin_idx(action)
        self.action_to_bin[bin_idx].append(self.idx)

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, f'buffer/{self.last_save}_{self.idx}.pt')
        action_to_idx_path = os.path.join(save_dir, f'act2idx/{self.last_save}_{self.idx}.pkl')
        payload = [
            self.obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)
        with open(action_to_idx_path, 'wb') as wrt:
            pickle.dump(self.action_to_bin, wrt)

    def load(self, save_dir):
        chunks = os.listdir(os.path.join(save_dir, 'buffer'))
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, 'buffer', chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.actions[start:end] = payload[1]
            self.rewards[start:end] = payload[2]
            self.next_obses[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

            act2idx_path = path.replace('buffer', 'act2idx').replace('pt', 'pkl')
            with open(act2idx_path, 'rb') as rd:
                self.action_to_bin.update(
                    pickle.load(rd)
                )
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_of_rollout_steps', default=100, type=int)
    parser.add_argument('--num_of_training_steps', default=100, type=int)
    args = parser.parse_args()
    return args

def warmup_replaybuffer(work_dir):
    args = parse_args()

    env = dmc2gym.make(
        domain_name="cartpole",
        task_name="swingup",
        seed=args.seed,
        visualize_reward=False,
        from_pixels="rssm",
        height=100,
        width=100,
        frame_skip=8
    )
    env.seed = args.seed

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(args.num_of_rollout_steps, obs_shape, act_shape, 24, device)

    obs = env.reset()
    episode, episode_reward, done = 0, 0, False
    for step in range(args.num_of_rollout_steps):
        if done:
            obs = env.reset()
            episode_step, episode_reward, done = 0, 0, False
            episode += 1
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.append(obs, action, reward, next_obs, done)
        obs = next_obs
        # print(step, reward)

    # for idx, lst in replay_buffer.action_to_bin.items():
        # print(idx, len(lst))
    replay_buffer.save(work_dir)

def train_bisim_net(work_dir):
    args = parse_args()

    env = dmc2gym.make(
        domain_name="cartpole",
        task_name="swingup",
        seed=args.seed,
        visualize_reward=False,
        from_pixels="rssm",
        height=100,
        width=100,
        frame_skip=8
    )
    env.seed = args.seed

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    model = BisimMetric(obs_shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(1000, obs_shape, act_shape, 24, device)
    replay_buffer.load(work_dir)

    gamma = 0.9 # bisimulation parameter
    optimizer = Adam(model.parameters(), lr=1e-3)

    for i in range(args.num_of_training_steps):
        action = env.action_space.sample()
        rst = replay_buffer.get_other_pairs(action)
        if rst is None:
            continue
        obses, actions, rewards, next_obses, not_dones = rst

        batch_size = obses.size(0)
        perm = np.random.permutation(batch_size)
        obses2 = obses[perm]
        rewards2 = rewards[perm]
        next_obses2 = next_obses[perm]

        t = torch.abs(rewards - rewards2) + gamma * model(next_obses, next_obses2)
        pred = model(obses, obses2)
        t = torch.max(t, pred) 
        loss = ((t - pred) * (t - pred)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(i, loss)

    params = dict(bisim=model)
    torch.save(params, f'{work_dir}/model/bisim.pt')


def load_model():
    model = torch.load(f'exp/bisim/model/bisim.pt')['bisim']

    args = parse_args()

    env = dmc2gym.make(
        domain_name="cartpole",
        task_name="swingup",
        seed=args.seed,
        visualize_reward=False,
        from_pixels="rssm",
        height=100,
        width=100,
        frame_skip=8
    )
    env.seed = args.seed

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(1000, obs_shape, act_shape, 24, device)
    replay_buffer.load('exp/bisim')

    action = env.action_space.sample()
    rst = replay_buffer.get_other_pairs(action=action)
    if rst is None:
        print("Action miss")
        return
    obses, actions, rewards, next_obses, not_dones = rst
    batch_size = obses.size(0)
    perm = np.random.permutation(batch_size)
    obses2 = obses[perm]
    bisim_dist = model(obses, obses2)
    # print(bisim_dist)


def main():
    work_dir = 'exp/bisim'
    if not os.path.isdir(work_dir):
        os.system(f'mkdir {work_dir}')
    if not os.path.isdir(work_dir + '/buffer'):
        os.system('mkdir {}/buffer {}/act2idx {}/model'.format(work_dir, work_dir, work_dir))
    warmup_replaybuffer(work_dir)
    train_bisim_net(work_dir)



if __name__ == "__main__":
    main()
