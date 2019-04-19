import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time

import torch
import torch.nn as nn

import logger
import utils
from SAC import SAC
from DistSAC import DistSAC


def render_env(env):
    try:
        return env.render(mode='rgb_array', width=256, height=256, camera_id=0).copy()
    except:
        pass
    return env.render(mode='rgb_array').copy()
    

def evaluate_policy(env,
                    policy,
                    tracker,
                    total_timesteps,
                    save_dir,
                    num_episodes=10,
                    render=True):
    if render:
        video_dir = utils.make_dir(os.path.join(save_dir, 'video'))
    tracker.reset('eval_episode_reward')
    tracker.reset('eval_episode_timesteps')
    for i in range(num_episodes):

        state = env.reset()
        if render and i == 0:
            frames = [render_env(env)]
        done = False
        sum_reward = 0
        timesteps = 0
        while not done:
            with torch.no_grad():
                action = policy.select_action(
                    torch.FloatTensor(state), update_stat=False)
            state, reward, done, _ = env.step(action)
            if render and i == 0:
                frames.append(render_env(env))
            sum_reward += reward
            timesteps += 1

        if render and i == 0:
            frames = [
                torch.tensor(frame.copy()).float() / 255 for frame in frames
            ]
            file_name = os.path.join(video_dir, '%d.mp4' % total_timesteps)
            utils.save_gif(file_name, frames, color_last=True)

        tracker.update('eval_episode_reward', sum_reward)
        tracker.update('eval_episode_timesteps', timesteps)
        

def calc_max_episode_steps(env, dmc):
    if hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps
    if hasattr(env, 'env') and hasattr(env.env, '_max_episode_steps'):
        return env.env._max_episode_steps
    return 1000 if dmc else 10000


def make_env(args):
    if args.dmc:
        import dm_control2gym
        env = dm_control2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            task_kwargs={'random': args.seed},
            visualize_reward=True)
    else:
        from pointmass import point_mass
        env = gym.make(args.env_name)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='HalfCheetah-v2')
    parser.add_argument('--domain_name', default='point_mass')
    parser.add_argument('--task_name', default='easy')
    parser.add_argument('--start_timesteps', default=10000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--max_timesteps', default=1e6, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--replay_buffer_size', default=1000000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--initial_temperature', default=0.01, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--log_format', default='json', type=str)
    parser.add_argument('--save_dir', default='.', type=str)
    parser.add_argument('--dmc', default=False, action='store_true')
    parser.add_argument('--no_eval_save', default=False, action='store_true')
    parser.add_argument('--no_render', default=False, action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--expl_coef', default=0., type=float)
    parser.add_argument('--num_candidates', default=1, type=int)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = make_env(args)

    # Set seeds
    utils.set_seed_everywhere(args.seed)
    env.seed(args.seed)
    
    utils.make_dir(os.path.join(args.save_dir, 'model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = calc_max_episode_steps(env, args.dmc)

    replay_buffer = utils.ReplayBuffer(args.replay_buffer_size)
    state_buffer = utils.StateBuffer(1000)
    
    
    dist_policy = DistSAC(device, state_dim, action_dim, max_action, args.initial_temperature, args.lr, args.num_candidates)
    policy = SAC(device, state_dim, action_dim, max_action,
                 args.initial_temperature, args.lr)
     
    tracker = logger.StatsTracker()
    train_logger = logger.TrainLogger(
        args.log_format, file_name=os.path.join(args.save_dir, 'train.log'))
    eval_logger = logger.EvalLogger(
        args.log_format, file_name=os.path.join(args.save_dir, 'eval.log'))

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = True

    
    evaluate_policy(
        env,
        policy,
        tracker,
        total_timesteps,
        args.save_dir,
        num_episodes=args.num_eval_episodes,
        render=not args.no_render)
    eval_logger.dump(tracker)

    start_time = time.time()
    while total_timesteps < args.max_timesteps:

        if done:
            if total_timesteps != 0:
                tracker.update('fps', episode_timesteps / (time.time() - start_time))
                start_time = time.time()
                train_logger.dump(tracker)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluate_policy(
                    env,
                    policy,
                    tracker,
                    total_timesteps,
                    args.save_dir,
                    num_episodes=args.num_eval_episodes,
                    render=not args.no_render)
                eval_logger.dump(tracker)

                if not args.no_eval_save:
                    policy.save(args.save_dir, total_timesteps)
                    dist_policy.save(args.save_dir, total_timesteps)

            tracker.update('train_episode_reward', episode_reward)
            tracker.update('train_episode_timesteps', episode_timesteps)
            # Reset environment
            if args.domain_name == 'point_mass':
                while True:
                    state = env.reset()
                    if state[0] > 0 and state[1] > 0:
                        break
            else:
                state = env.reset()
             
            state_buffer.add(state)
            goal_states = torch.FloatTensor(state_buffer.get()).to(device)
            torch.save(goal_states, os.path.join(args.save_dir, 'goals.pt'))
                
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            tracker.update('num_episodes')
            tracker.reset('episode_timesteps')

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy.sample_action(np.array(state))
            

        if total_timesteps >= 1e3:
            dist_policy.train(replay_buffer, total_timesteps, tracker,
                                  args.batch_size, args.discount, args.tau,
                                  args.policy_freq, target_entropy=-action_dim)

            
            policy.train(replay_buffer, total_timesteps, tracker,
                         args.batch_size, args.discount, args.tau,
                         args.policy_freq, target_entropy=-action_dim)
            
            
        new_state, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
        episode_reward += reward
        
        if args.expl_coef > 0:
            with torch.no_grad():
                expl_bonus, _ = dist_policy.get_distance(
                    torch.FloatTensor(state).to(device), goal_states)
                expl_bonus *= args.expl_coef
            tracker.update('expl_bonus', expl_bonus)
            reward += expl_bonus

        replay_buffer.add(state, action, reward, new_state, done_bool, done)

        state = new_state

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        tracker.update('total_timesteps')
        tracker.update('episode_timesteps')

    policy.save(args.save_dir, total_timesteps)
    dist_policy.save(args.save_dir, total_timesteps)


if __name__ == '__main__':
    main()
