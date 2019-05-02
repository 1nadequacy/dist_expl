import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import cv2

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from logger import Logger
import utils
from SAC import SAC
from DistSAC import DistSAC
from GoalSAC import GoalSAC


def render_env(env):
    try:
        return env.render(
            mode='rgb_array', width=256, height=256, camera_id=0).copy()
    except:
        pass
    return env.render(mode='rgb_array').copy()


def calc_point_xy(env_name, point, shape):
    if env_name.startswith('AntMaze'):
        x1, x2, y1, y2 = -16, 32, 32, -16
    elif env_name.startswith('AntPush'):
        x1, x2, y1, y2 = -24, 24, 32, -16
    elif env_name.startswith('AntFall'):
        x1, x2, y1, y2 = -19, 29, 29, -13
    
    x = int((point[0] - x1) / (x2 - x1) * shape[0])
    y = int((point[1] - y1) / (y2 - y1) * shape[1])
    return x, y
       


def evaluate_policy(env,
                    args,
                    policy,
                    dist_policy,
                    L,
                    step):
    if not args.no_render:
        video_dir = utils.make_dir(os.path.join(args.save_dir, 'video'))
    for i in range(args.num_eval_episodes):
        state = reset_env(env, args, eval_mode=True)
        if not args.no_render and i == 0:
            frames = [render_env(env)]
        done = False
        sum_reward = 0
        expl_bonus = 0
        last_add = 0
        timesteps = 0
        ctx_buffer = utils.ContextBuffer(args.ctx_size, state.shape[0])
        while not done:
            ctx = ctx_buffer.get()
            with torch.no_grad():
                action = policy.select_action(state, ctx)
                if args.expl_coef > 0:
                    dist, _ = dist_policy.get_distance_numpy(state, ctx)
                    if dist.sum().item() > args.dist_threshold or timesteps - last_add > args.max_gap:
                        ctx_buffer.add(state)
                        expl_bonus += args.expl_coef
                        last_add = timesteps

            state, reward, done, _ = env.step(action)
            if not args.no_render and i == 0:
                frames.append(render_env(env))
                if args.env_type == 'ant':
                    for point in ctx_buffer.storage:
                        x, y = calc_point_xy(args.env_name, point, frames[-1].shape)
                        cv2.circle(frames[-1], (x, y), 1, (255, 255, 0), 5)
            sum_reward += reward
            timesteps += 1

        if not args.no_render and i == 0:
            frames = [
                torch.tensor(frame.copy()).float() / 255 for frame in frames
            ]
            file_name = os.path.join(video_dir, '%d.mp4' % step)
            utils.save_gif(file_name, frames, color_last=True)

        if args.env_type == 'ant':
            L.log('eval/episode_success', env.get_success(reward), step)
        L.log('eval/episode_reward', sum_reward, step)
        L.log('eval/episode_expl_bonus', expl_bonus, step)


def calc_max_episode_steps(env, env_type):
    if hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps
    if hasattr(env, 'env') and hasattr(env.env, '_max_episode_steps'):
        return env.env._max_episode_steps
    if env_type == 'dmc':
        return 1000
    return 100000


def make_env(args):
    if args.env_type == 'dmc':
        import dm_control2gym
        env = dm_control2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            task_kwargs={'random': args.seed},
            visualize_reward=True)
    elif args.env_type == 'mjc':
        from pointmass import point_mass
        env = gym.make(args.env_name)
    elif args.env_type == 'ant':
        import ant_env_mujoco
        env = gym.make(args.env_name)
    else:
        assert 'unknown env type: %s' % args.env_type
    return env


def reset_env(env, args, eval_mode=False):
    if args.env_type == 'ant':
        return env.reset(eval_mode=eval_mode)
    if args.env_type != 'dmc' or args.domain_name != 'point_mass':
        return env.reset()

    while True:
        state = env.reset()
        if abs(state[0]) > 0.25 or abs(state[1]) > 0.25:
            #if abs(state[0]) < 0.05 and abs(state[1]) < 0.05:
            break
    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='HalfCheetah-v2')
    parser.add_argument('--domain_name', default='point_mass')
    parser.add_argument('--task_name', default='easy')
    parser.add_argument('--start_timesteps', default=10000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--max_timesteps', default=1e6, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--replay_buffer_size', default=1000000, type=int)
    parser.add_argument('--ctx_size', default=0, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--initial_temperature', default=0.01, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--log_format', default='json', type=str)
    parser.add_argument('--save_dir', default='.', type=str)
    parser.add_argument('--env_type', default='dmc', type=str)
    parser.add_argument('--no_eval_save', default=False, action='store_true')
    parser.add_argument('--no_render', default=False, action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--expl_coef', default=0., type=float)
    parser.add_argument('--dist_threshold', default=10, type=float)
    parser.add_argument('--use_l2', default=0, type=int)
    parser.add_argument('--only_pos', default=0, type=int)
    parser.add_argument('--num_candidates', default=1, type=int)
    parser.add_argument('--max_gap', default=1000, type=int)
    parser.add_argument('--use_tb', default=False, action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    
    env = make_env(args)
    env.seed(args.seed)
    
    utils.make_dir(args.save_dir)
    utils.make_dir(os.path.join(args.save_dir, 'model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #sw = SummaryWriter(os.path.join(args.save_dir, 'summary'), comment='aaa')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    max_episode_steps = calc_max_episode_steps(env, args.env_type)

    replay_buffer = utils.ReplayBuffer(args.replay_buffer_size,
                                       args.ctx_size)

    dist_policy = DistSAC(
        device,
        state_dim,
        action_dim,
        args.initial_temperature,
        args.lr,
        args.num_candidates,
        use_l2=args.use_l2,
        only_pos=args.only_pos)
    policy_model = GoalSAC if args.ctx_size > 0 else SAC
    policy = policy_model(
        device,
        state_dim,
        action_dim,
        args.initial_temperature,
        args.lr,
        ctx_size=args.ctx_size)

    L = Logger(args.save_dir, use_tb=args.use_tb)
   
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_expl_bonus = 0
    episode_timesteps = 0
    done = True

    evaluate_policy(
        env,
        args,
        policy,
        dist_policy,
        L,
        total_timesteps)
    L.dump(total_timesteps)

    start_time = time.time()
    while total_timesteps < args.max_timesteps:

        if done:
            if total_timesteps != 0:
                L.log('train/duration', time.time() - start_time, total_timesteps)
                start_time = time.time()
                L.dump(total_timesteps)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluate_policy(
                    env,
                    args,
                    policy,
                    dist_policy,
                    L,
                    total_timesteps)
                L.dump(total_timesteps)

                if not args.no_eval_save:
                    policy.save(args.save_dir, total_timesteps)
                    dist_policy.save(args.save_dir, total_timesteps)

            L.log('train/episode_reward', episode_reward, total_timesteps)
            L.log('train/episode_expl_bonus', episode_expl_bonus, total_timesteps)
            
            # Reset environment
            state = reset_env(env, args)

            ctx_buffer = utils.ContextBuffer(args.ctx_size, state_dim)

            done = False
            episode_reward = 0
            episode_expl_bonus = 0
            last_add = total_timesteps
            episode_timesteps = 0
            episode_num += 1

            L.log('train/episode', episode_num, total_timesteps)

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            ctx = ctx_buffer.get()

            with torch.no_grad():
                action = policy.sample_action(state, ctx)
                if args.expl_coef > 0:
                    dist, _ = dist_policy.get_distance_numpy(state, ctx)

                    if dist.sum().item() > args.dist_threshold or total_timesteps - last_add > args.max_gap:
                        ctx_buffer.add(state)
                        episode_expl_bonus += args.expl_coef
                        last_add = total_timesteps


        if total_timesteps >= 1e3 and args.expl_coef > 0 and args.use_l2 == 0:
            num_updates = int(1e3) if total_timesteps == 1e3 else 1
            for _ in range(num_updates):
                dist_policy.train(
                    replay_buffer,
                    total_timesteps,
                    L,
                    args.batch_size,
                    args.discount,
                    args.tau,
                    args.policy_freq,
                    target_entropy=-action_dim)

        if total_timesteps >= args.start_timesteps:
            num_updates = args.start_timesteps if total_timesteps == args.start_timesteps else 1
            for _ in range(num_updates):
                policy.train(
                    replay_buffer,
                    total_timesteps,
                    dist_policy,
                    L,
                    args.batch_size,
                    args.discount,
                    args.tau,
                    args.policy_freq,
                    target_entropy=-action_dim,
                    expl_coef=args.expl_coef,
                    dist_threshold=args.dist_threshold)

        new_state, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(
            done)
        episode_reward += reward

        replay_buffer.add(state, action, reward, new_state, done_bool, done)

        state = new_state

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    policy.save(args.save_dir, total_timesteps)
    dist_policy.save(args.save_dir, total_timesteps)


if __name__ == '__main__':
    main()
