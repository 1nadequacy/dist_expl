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
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(2 * state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)

        self.apply(weight_init)

    def forward(self, x, x_g, compute_pi=True, compute_log_pi=True):
        x = torch.cat([x, x_g], dim=1)
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
        self.l1 = nn.Linear(2 * state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(2 * state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.apply(weight_init)
        

    def forward(self, x, x_g, u):
        xu = torch.cat([x, x_g, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2
        


class DistSAC(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 max_action,
                 initial_temperature,
                 lr=1e-3,
                 num_candidates=1,
                 log_std_min=-20,
                 log_std_max=2):
        self.device = device
        
        self.num_candidates = num_candidates

        self.actor = Actor(state_dim, action_dim, max_action, log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.max_action = max_action
        

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_distance(self, state, repeated_goal_state, idxs=None):
        batch_size, num_goals, state_dim = repeated_goal_state.size()
        repeated_state = state.unsqueeze(1).repeat(1, num_goals, 1)
        
        repeated_state = repeated_state.view(-1, state_dim)
        repeated_goal_state = repeated_goal_state.view(-1, state_dim)
        
        repeated_action, _, _ = self.actor(repeated_state, repeated_goal_state, compute_pi=False, compute_log_pi=False)
        q1, q2 = self.critic(repeated_state, repeated_goal_state, repeated_action)
        dist = -torch.min(q1, q2)
        dist = dist.view(batch_size, num_goals)
        
        if idxs is None:
            num_candidates = min(self.num_candidates, num_goals)
            dist, idxs = dist.topk(num_candidates, dim=1, largest=False)
        else:
            dist = dist.gather(1, idxs)
        dist = dist.mean(dim=1, keepdim=True)
        
        return dist, idxs
    
    def train(self,
              replay_buffer,
              total_timesteps,
              tracker,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_freq=2,
              target_entropy=None):
        
        # Sample replay buffer
        state, action, _, next_state, _, goals = replay_buffer.sample(
                batch_size, with_goal=True)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        
        perm = torch.randperm(state.size(0))
        mask_noise = torch.rand(state.size(0), 1).to(self.device)
        random_goal_mask = (mask_noise < 0.1).float()
        same_goal_mask = (mask_noise > 0.9).float()
        traj_goal_mask = 1 - (random_goal_mask + same_goal_mask)

        assert ((random_goal_mask + same_goal_mask + traj_goal_mask) - 1.0).norm() < 1e-3
        assert (random_goal_mask * same_goal_mask * traj_goal_mask).norm() < 1e-3

        goals = state[perm] * random_goal_mask + state * same_goal_mask + goals * traj_goal_mask

        done = ((state - goals).norm(dim=-1, keepdim=True) < 1e-5).float()
        reward = -(1 - done) 
        
        tracker.update('dist_train_reward', reward.sum().item(), reward.size(0))
        

        def fit_critic():
            with torch.no_grad():
                _, policy_action, log_pi = self.actor(next_state, goals)
                target_Q1, target_Q2 = self.critic_target(
                    next_state, goals, policy_action)
                target_V = torch.min(target_Q1,
                                     target_Q2) - self.alpha.detach() * log_pi
                target_Q = reward + (1 - done) * discount * target_V

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, goals, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
            tracker.update('dist_critic_loss', critic_loss * current_Q1.size(0),
                           current_Q1.size(0))
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        fit_critic()

        def fit_actor():
            # Compute actor loss
            _, pi, log_pi = self.actor(state, goals)
            actor_Q1, actor_Q2 = self.critic(state, goals, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
            tracker.update('dist_actor_loss', actor_loss * state.size(0),
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
                   '%s/model/dist_actor_%s.pt' % (directory, timestep))
        torch.save(self.critic.state_dict(),
                   '%s/model/dist_critic_%s.pt' % (directory, timestep))

    def load(self, directory, timestep):
        self.actor.load_state_dict(
            torch.load('%s/model/dist_actor_%s.pt' % (directory, timestep)))
        self.critic.load_state_dict(
            torch.load('%s/model/dist_critic_%s.pt' % (directory, timestep)))
