from collections import OrderedDict
import json
import numpy as np


class IntegerMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0

    def update(self, n=1):
        self.n += n

    def compute(self):
        return self.n

    def __str__(self):
        return '%d' % self.compute()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def compute(self):
        return self.sum / max(1, self.count)

    def __str__(self):
        return '%.03f' % self.compute()


class MovingAverageMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.vals = np.zeros(window_size)
        self.counts = np.zeros(window_size)
        self.pointer = 0

    def reset(self):
        self.vals.fill(0)
        self.counts.fill(0)
        self.pointer = 0

    def update(self, val, n=1):
        self.vals[self.pointer] = val
        self.counts[self.pointer] = n
        self.pointer = (self.pointer + 1) % self.window_size

    def compute(self):
        return self.vals.sum() / max(1., self.counts.sum())

    def __str__(self):
        return '%.03f' % self.compute()


class StatsTracker(object):
    def __init__(self, window_size=1000):
        self.meters = OrderedDict()
        self.meters['train_episode_reward'] = AverageMeter()
        self.meters['train_episode_timesteps'] = AverageMeter()
        self.meters['eval_episode_reward'] = AverageMeter()
        self.meters['eval_episode_timesteps'] = AverageMeter()
        self.meters['train_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()

        self.meters['train_reward'] = MovingAverageMeter(window_size)
        self.meters['dist_train_reward'] = MovingAverageMeter(window_size)
        self.meters['train_predicted_reward'] = MovingAverageMeter(window_size)
        self.meters['reward_pearsonr'] = MovingAverageMeter(window_size)
        self.meters['actor_loss'] = MovingAverageMeter(window_size)
        self.meters['critic_loss'] = MovingAverageMeter(window_size)
        self.meters['dist_actor_loss'] = MovingAverageMeter(window_size)
        self.meters['dist_critic_loss'] = MovingAverageMeter(window_size)
        self.meters['policy_loss'] = MovingAverageMeter(window_size)
        self.meters['gail_loss'] = MovingAverageMeter(window_size)
        self.meters['flow_loss'] = MovingAverageMeter(window_size)
        self.meters['pot_loss'] = MovingAverageMeter(window_size)
        self.meters['pot_diff'] = MovingAverageMeter(window_size)
        self.meters['gail_reward'] = MovingAverageMeter(window_size)
        self.meters['pot_coef'] = AverageMeter()
        self.meters['expl_bonus'] = MovingAverageMeter(window_size)

        self.meters['total_timesteps'] = IntegerMeter()
        self.meters['num_target_states'] = AverageMeter()
        self.meters['fps'] = AverageMeter()
        self.meters['num_episodes'] = IntegerMeter()
        self.meters['episode_timesteps'] = IntegerMeter()
        self.meters['epoch'] = IntegerMeter()

    def reset(self, name=None):
        if name is None:
            for name in self.meters:
                self.meters[name].reset()
        else:
            self.meters[name].reset()

    def update(self, name, *args):
        self.meters[name].update(*args)


class Logger(object):
    def __init__(self, format_type, file_name=None, keys=[]):
        self.format_type = format_type
        self.keys = keys
        self.log_type = None
        self.log_file = open(file_name, 'w') if file_name is not None else None

    def _format_json(self, stats):
        return json.dumps(stats)

    def _format_text(self, stats):
        pieces = []
        for key in stats:
            pieces.append('%s: %s' % (key, stats[key]))
        return '| ' + (' | '.join(pieces))

    def _format(self, stats):
        if self.format_type == 'json':
            return self._format_json(stats)
        elif self.format_type == 'text':
            return self._format_text(stats)
        assert False, 'unknown log_type: %s' % self.format_type

    def dump(self, tracker):
        stats = OrderedDict()
        stats['type'] = self.log_type
        for key in self.keys:
            stats[key] = str(tracker.meters[key])
        print(self._format(stats), flush=True)
        if self.log_file is not None:
            print(self._format(stats), flush=True, file=self.log_file)


class TrainLogger(Logger):
    def __init__(self, format_type, file_name=None, init_keys=None):
        keys = init_keys or [
            'total_timesteps',
            'num_episodes',
            'num_target_states',
            'fps',
            'episode_timesteps',
            'train_episode_reward',
            'train_episode_timesteps',
            'train_reward',
            'dist_train_reward',
            'actor_loss',
            'critic_loss',
            'dist_actor_loss',
            'dist_critic_loss',
            'pot_diff',
            'gail_reward',
            'pot_coef',
            'train_predicted_reward',
            'reward_pearsonr',
            'gail_loss',
            'expl_bonus',
        ]
        super(TrainLogger, self).__init__(format_type, file_name, keys)
        self.log_type = 'train'


class EvalLogger(Logger):
    def __init__(self, format_type, file_name=None, init_keys=None):
        keys = init_keys or [
            'total_timesteps',
            'num_episodes',
            'num_target_states',
            'fps',
            'episode_timesteps',
            'eval_episode_reward',
            'eval_episode_timesteps',
        ]
        super(EvalLogger, self).__init__(format_type, file_name, keys)
        self.log_type = 'eval'
