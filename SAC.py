import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import utils


EPS = 1e-8

# From https://github.com/openai/spinningup/blob/master/spinup/algos/sac/core.py


def gaussian_likelihood(noise, log_std):
    pre_sum = -0.5 * noise.pow(2) - log_std
    return pre_sum.sum(
        -1, keepdim=True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def apply_squashing_func(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)

        self.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu, log_std = self.l3(x).chunk(2, dim=-1)
        
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1)

        
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_likelihood(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)

        return mu, pi, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


class SAC(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 initial_temperature,
                 lr=1e-3,
                 log_std_min=-20,
                 log_std_max=2,
                 ctx_size=0):
        self.device = device
        assert ctx_size == 0

        self.actor = Actor(state_dim, action_dim, log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)


    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, ctx):
        # ctx should be unused
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            mu, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state, ctx):
        # ctx should be unused
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            mu, pi, _ = self.actor(state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()
        
    
    def get_value(self, state, num_samples=5):
        targets = []
        for _ in range(num_samples):
            with torch.no_grad():
                _, action, _ = self.actor(state, compute_log_pi=False)
                target_Q1, target_Q2 = self.critic(state, action)
            targets.append(torch.min(target_Q1, target_Q2))
        target = torch.cat(targets, dim=1).mean(dim=1)
        return target
            
    def train(self,
              replay_buffer,
              step,
              dist_policy,
              L,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_freq=2,
              target_entropy=None,
              expl_coef=0.0,
              dist_threshold=10):
        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(
                batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).view(-1, 1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device).view(-1, 1)
        
        if expl_coef > 0:
            ctx = torch.zeros_like(state).to(self.device)
            ctx = ctx.unsqueeze(1)
            with torch.no_grad():
                dist, _ = dist_policy.get_distance(state, ctx)
            expl_bonus = dist * expl_coef
            L.log('train/expl_bonus', expl_bonus.sum().item(), step, n=expl_bonus.size(0))
            reward += expl_bonus.detach()
            
        
        
        L.log('train/batch_reward', reward.sum().item(), reward.size(0), step)

        def fit_critic():
            with torch.no_grad():
                _, policy_action, log_pi = self.actor(next_state)
                target_Q1, target_Q2 = self.critic_target(
                    next_state, policy_action)
                target_V = torch.min(target_Q1,
                                     target_Q2) - self.alpha.detach() * log_pi
                target_Q = reward + (done * discount * target_V)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
            L.log('train/critic_loss', critic_loss.item() * current_Q1.size(0),
                           step, n=current_Q1.size(0))
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        fit_critic()

        def fit_actor():
            # Compute actor loss
            _, pi, log_pi = self.actor(state)
            actor_Q1, actor_Q2 = self.critic(state, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
            L.log('train/actor_loss', actor_loss.item() * state.size(0),
                           step, n=state.size(0))
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if target_entropy is not None:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (
                    self.alpha * (-log_pi - target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

        if step % policy_freq == 0:
            fit_actor()
            
            utils.soft_update_params(self.critic, self.critic_target, tau)

    def save(self, directory, timestep):
        torch.save(self.actor.state_dict(),
                   '%s/model/actor_%s.pt' % (directory, timestep))
        torch.save(self.critic.state_dict(),
                   '%s/model/critic_%s.pt' % (directory, timestep))

    def load(self, directory, timestep):
        self.actor.load_state_dict(
            torch.load('%s/model/actor_%s.pt' % (directory, timestep)))
        self.critic.load_state_dict(
            torch.load('%s/model/critic_%s.pt' % (directory, timestep)))
