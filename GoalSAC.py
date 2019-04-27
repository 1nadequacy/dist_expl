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

        
class Encoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        
        
        # bz X n X 2 * state
        self.trunk = nn.Sequential(
            nn.Linear(2 * state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256))
        
    def forward(self, x, c):
        # x: bsz X state
        # c: bsz X n X state
        x = x.unsqueeze(1).expand(x.size(0), c.size(1), x.size(1))
        xc = torch.cat([x, c], dim=2)
        
        h = self.trunk(xc)
        h = h.mean(dim=1)
        h = F.relu(h)
        # h: bsz X 32
        return h
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.encoder = Encoder(state_dim)
        self.actor = nn.Linear(256, 2 * action_dim)
        

        self.apply(weight_init)

    def forward(self, x, c, compute_pi=True, compute_log_pi=True):
        h = self.encoder(x, c)
        
        mu, log_std = self.actor(h).chunk(2, dim=-1)
        
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
        
        self.encoder = Encoder(state_dim)

        # Q1 architecture
        self.l1 = nn.Linear(256 + action_dim, 256)
        self.l2 = nn.Linear(256, 1)

        # Q2 architecture
        self.l3 = nn.Linear(256 + action_dim, 256)
        self.l4 = nn.Linear(256, 1)

        self.apply(weight_init)
        

    def forward(self, x, c, u):
        h = self.encoder(x, c)
        
        hu = torch.cat([h, u], dim=1)

        x1 = F.relu(self.l1(hu))
        x1 = self.l2(x1)

        x2 = F.relu(self.l3(hu))
        x2 = self.l4(x2)
        return x1, x2
                
        

class GoalSAC(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 initial_temperature,
                 lr=1e-3,
                 log_std_min=-20,
                 log_std_max=2,
                 ctx_size=10):
        self.device = device

        self.actor = Actor(state_dim, action_dim, log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.ctx_size = ctx_size
        

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action(self, state, ctx):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ctx = torch.FloatTensor(ctx).unsqueeze(0).to(self.device)
            mu, _, _ = self.actor(
                state, ctx, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state, ctx):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ctx = torch.FloatTensor(ctx).unsqueeze(0).to(self.device)
            mu, pi, _ = self.actor(state, ctx, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()
        
    
    def get_value(self, state, ctx, num_samples=5):
        targets = []
        for _ in range(num_samples):
            with torch.no_grad():
                _, action, _ = self.actor(state, ctx, compute_log_pi=False)
                target_Q1, target_Q2 = self.critic(state, ctx, action)
            targets.append(torch.min(target_Q1, target_Q2))
        target = torch.cat(targets, dim=1).mean(dim=1)
        return target
    
    def update_context(self, state, ctx, mask):
        # state: BxS
        # ctx: BxKxS
        # mask: Bx1
        if ctx.size(1) >= self.ctx_size:
            
            mask = mask.unsqueeze(1).expand_as(ctx).contiguous()
            mask[:, 1:, :].fill_(0)
            state = state.unsqueeze(1).expand_as(ctx)

            next_ctx = ctx * (1 - mask) + state * mask
            
        else:
            next_ctx = torch.cat([ctx, state.unsqueeze(1)], dim=1)
        return next_ctx
    
            
    def train(self,
              replay_buffer,
              total_timesteps,
              dist_policy,
              tracker,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_freq=2,
              target_entropy=None,
              expl_coef=0.0,
              dist_threshold=10):
        # Sample replay buffer
        state, action, reward, next_state, done, ctx = replay_buffer.sample(
                batch_size, with_ctx=True)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).view(-1, 1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device).view(-1, 1)
        ctx = torch.FloatTensor(ctx).to(self.device)
        
        assert expl_coef > 0
        
        with torch.no_grad():
            dist, _ = dist_policy.get_distance(state, ctx)
        novel_mask = (dist > dist_threshold).float()
        expl_bonus = novel_mask * expl_coef
        tracker.update('expl_bonus', expl_bonus.sum().item(), expl_bonus.size(0))
        next_ctx = self.update_context(state, ctx, novel_mask)

        reward += expl_bonus.detach()
        
            
        
        tracker.update('train_reward', reward.sum().item(), reward.size(0))

        def fit_critic():
            with torch.no_grad():
                _, policy_action, log_pi = self.actor(next_state, next_ctx)
                target_Q1, target_Q2 = self.critic_target(
                    next_state, next_ctx, policy_action)
                target_V = torch.min(target_Q1,
                                     target_Q2) - self.alpha.detach() * log_pi
                target_Q = reward + (done * discount * target_V)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, ctx, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
            tracker.update('critic_loss', critic_loss * current_Q1.size(0),
                           current_Q1.size(0))
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        fit_critic()

        def fit_actor():
            # Compute actor loss
            _, pi, log_pi = self.actor(state, ctx)
            actor_Q1, actor_Q2 = self.critic(state, ctx, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
            tracker.update('actor_loss', actor_loss * state.size(0),
                           state.size(0))
            
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

        if total_timesteps % policy_freq == 0:
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
