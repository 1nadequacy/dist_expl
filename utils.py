import torch
import numpy as np
import torch.nn as nn
import os
import imageio
import random
from collections import deque


class eval_mode(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.prev = self.model.training
        self.model.train(False)

    def __exit__(self, *args):
        self.model.train(self.prev)
        return False
    
    
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def weight_reset(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
        
        
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(),
                                target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                            (1 - tau) * target_param.data)
        
        
def soft_update_buffers(net, target_net, tau):
    for buff, target_buff in zip(net.buffers(),
                                target_net.buffers()):
            target_buff.data.copy_(tau * buff.data +
                            (1 - tau) * target_buff.data)
                


def save_gif(filename, inputs, bounce=False, color_last=False, duration=0.1):
    images = []
    for tensor in inputs:
        tensor = tensor.cpu()
        if not color_last:
            tensor = tensor.transpose(0,1).transpose(1,2)
        tensor = tensor.clamp(0,1)
        images.append((tensor.numpy() * 255).astype('uint8'))
    if bounce:
        images = images + list(reversed(images[1:-1]))
    imageio.mimsave(filename, images)
    

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class NormReplayBuffer(object):
    def __init__(self, max_size=1e6, norm_ret=False, discount=0.99, alpha=0.001):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
        self.norm_ret = norm_ret
        if norm_ret:
            self.norm_ret = norm_ret
            self.discount = discount
            self.alpha = alpha
            self.returns = 0.0
            self.returns_ema = None
            self.returns_ema_var = None
        
    def _update_ema_var(self, reward, mask):
        # From https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
        self.returns = self.returns * self.discount * mask + reward

        if self.returns_ema is None:
            self.returns_ema = self.returns
            self.returns_ema_var = 0
        else:
            delta = self.returns - self.returns_ema
            self.returns_ema += self.alpha * delta
            self.returns_ema_var = (1 - self.alpha) * (self.returns_ema_var + self.alpha * (delta ** 2))
        
        
    def add(self, data):
        if self.norm_ret:
            reward = data[-2]
            mask = 1 - data[-1]
            self._update_ema_var(reward, mask)
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            
        rewards = np.array(r).reshape(-1, 1)

        if self.norm_ret:
            rewards /= np.sqrt(self.returns_ema_var + 1e-8)
            # import ipdb; ipdb.set_trace()
            
        return np.array(x), np.array(y), np.array(u), rewards, np.array(d).reshape(-1, 1)
    
    
    
class ContextBuffer(object):
    def __init__(self, max_size, state_dim):
        self.storage = []
        self.max_size = max_size
        self.state_dim = state_dim
        
    def add(self, state):
        if len(self.storage) >= self.max_size:
            idx = np.random.randint(0, len(self.storage))
            self.storage[idx] = state
        else:
            self.storage.append(state)
        
    def __len__(self):
        return len(self.storage)
        
    def get(self):
        if len(self.storage) == 0:
            return np.zeros((1, self.state_dim))
        return np.array(self.storage)


class ReplayBuffer(object):
    def __init__(self, size, ctx_size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = deque()
        self._storage.append([])
        self._maxsize = size
        self._ctxsize = ctx_size
        self._currentsize = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, true_done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._currentsize >= self._maxsize:
            self._currentsize -= len(self._storage[0])
            self._storage.popleft()
        
        self._storage[-1].append(data)
        self._currentsize += 1
        
        if true_done:
            self._storage.append([])
            
        
    def _encode_sample(self, idxes, with_goal, with_ctx):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        if with_goal:
            goals = []
        if with_ctx:
            contexts = []
            
        ctx_size = np.random.randint(1, self._ctxsize + 1)
        
        for traj_idx, pos_idx in idxes:
            data = self._storage[traj_idx][pos_idx]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            
            if with_goal:
                goal_pos_idx = random.randint(pos_idx, len(self._storage[traj_idx]) - 1)
                obs_t = self._storage[traj_idx][goal_pos_idx][0]
                goals.append(np.array(obs_t, copy=False))
            if with_ctx:
                
                if pos_idx == 0:
                    ctxs = [np.zeros_like(obs_t) for _ in range(ctx_size)]
                else:
                    ctx_idxs = np.random.randint(0, pos_idx, size=(ctx_size,))
                    ctxs = []
                    for ctx_idx in ctx_idxs:
                        ctxs.append(self._storage[traj_idx][ctx_idx][0])
                contexts.append(np.array(ctxs))
            
        if with_goal:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(goals)
        elif with_ctx:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(contexts)
        else:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, with_goal=False, with_ctx=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        assert not (with_goal and with_ctx)
        
        traj_idxes = [random.randint(0, len(self._storage) - int(len(self._storage[-1]) == 0) - 1) for _ in range(batch_size)]
        pos_idxes = [random.randint(0, len(self._storage[idx]) - 1) for idx in traj_idxes]

        idxes = zip(traj_idxes, pos_idxes)
        return self._encode_sample(idxes, with_goal, with_ctx)

    
    
    
    
    
class Normalizer(nn.Module):
    def __init__(self, shape=()):
        super(Normalizer, self).__init__()
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.count = 1000
        
    def forward(self, x):
        x = (x - self.mean.type(x.type())) / (self.var.type(x.type()) + 1e-8).sqrt()
        return x.clamp(-5, 5)
        
        
    def update(self, x):
        x = x.type(torch.float64)
        batch_mean = x.mean(0)
        if x.size(0) > 1:
            batch_var = x.var(0)
        else:
            batch_var = x[0] * 0.0
        batch_count = x.size(0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        mean_, var_, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self.mean.copy_(mean_)
        self.var.copy_(var_)
        
        
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta * delta * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

