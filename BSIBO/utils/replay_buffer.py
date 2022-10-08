import torch
import numpy as np
import os
from collections import deque, namedtuple
import pickle
from torch.utils.data import Dataset
from collections import defaultdict
import random

from BSIBO.utils.utils import random_crop


class SuperReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(
        self, obs_shape, action_shape, capacity, batch_size, device,
        path_len=None, image_size=84, transform=None
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self._path_len = path_len


        self.action_to_bin = defaultdict(list)

    def _get_random_action(self):
        if self.idx == 0:
            return self.actions[0]
        idx = random.randint(0, self.idx - 1)
        return self.actions[idx]

    def _action_to_bin_idx(self, action):
        # action is np.ndarray
        # TODO: maybe too empirical
        res = (10 * (action + 1)).astype('int32')
        return ''.join(map(str, res))

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        # update action_to_bin
        bin_idx = self._action_to_bin_idx(action)
        self.action_to_bin[bin_idx].append(self.idx)


    def sample_proprio_without_crop(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        obses = self.obses[idxs]
        obses_crop = random_crop(obses, self.image_size)  # random crop takes numpy as input

        obses = torch.as_tensor(obses, device=self.device).float()
        obses_crop = torch.as_tensor(obses_crop, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, obses_crop, actions, rewards, next_obses, not_dones

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        # add random crop to pass observation into encoder to update using dbc loss
        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):
        # start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def _sample_sequential_idx(self, n, L):
        # Returns an index for a valid single chunk uniformly sampled from the
        # memory
        idx = np.random.randint(
            0, self.capacity - L if self.full else self.idx - L, size=n
        )
        pos_in_path = idx - idx // self._path_len * self._path_len
        idx[pos_in_path > self._path_len - L] = idx[
            pos_in_path > self._path_len - L
        ] // self._path_len * self._path_len + L
        idxs = np.zeros((n, L), dtype=np.int32)
        for i in range(n):
            idxs[i] = np.arange(idx[i], idx[i] + L)
        return idxs.transpose().reshape(-1)

    def sample_multi_view(self, n, L):
        # start = time.time()
        idxs = self._sample_sequential_idx(n, L)
        obses = self.obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, out=self.image_size)
        pos = random_crop(pos, out=self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()\
            .reshape(L, n, *obses.shape[-3:])
        actions = torch.as_tensor(self.actions[idxs], device=self.device)\
            .reshape(L, n, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)\
            .reshape(L, n)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)\
            .reshape(L, n)

        pos = torch.as_tensor(pos, device=self.device).float()\
            .reshape(L, n, *obses.shape[-3:])
        mib_kwargs = dict(view1=obses, view2=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, not_dones, mib_kwargs

    def sample_by_action(self):
        """
        Continuous case
        """
        action = self._get_random_action()
        bin_idx = self._action_to_bin_idx(action)
        if self.action_to_bin.get(bin_idx, None) is None:
            return None
        idx_pool = self.action_to_bin[bin_idx]
        idxs = np.random.choice(idx_pool, min(len(idx_pool), self.batch_size))

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()
        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir, act2idx_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        act2idx_path = os.path.join(act2idx_dir, '%d_%d.pkl' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)
        with open(act2idx_path, 'wb') as wrt:
            pickle.dump(self.action_to_bin, wrt)


    def load(self, save_dir, act2idx_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

        act2idx_chunks = sorted(os.listdir(act2idx_dir), lambda x: int(x.split('_')[0]))
        for act2idx_chunk in act2idx_chunks:
            act2idx_path = os.path.join(act2idx_dir, act2idx_chunk)
            with open(act2idx_path, 'rb') as rd:
                self.action_to_bin.update(
                    pickle.load(rd)
                )

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity
