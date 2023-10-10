import numpy as np

from .atari_env import OneHotAction, TimeLimit, Collect, RewardObs
from .atari_env import Atari
from .crafter import Crafter
from .tools import count_episodes, save_episodes, video_summary
import pathlib
import pdb
import json

def count_steps(datadir, cfg):
  return tools.count_episodes(datadir)[1]

def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.env.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'{prefix}/episodes', episodes)]
  step = count_steps(datadir, config)
  env_step = step * config.env.action_repeat
  with (pathlib.Path(config.logdir) / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', env_step)] + metrics)) + '\n')
  [writer.add_scalar('sim/' + k, v, env_step) for k, v in metrics]
  tools.video_summary(writer, f'sim/{prefix}/video', episode['image'][None, :1000], env_step)

  if 'episode_done' in episode:
    episode_done = episode['episode_done']
    num_episodes = sum(episode_done)
    writer.add_scalar(f'sim/{prefix}/num_episodes', num_episodes, env_step)
    # compute sub-episode len
    episode_done = np.insert(episode_done, 0, 0)
    episode_len_ = np.where(episode_done)[0]
    if len(episode_len_) > 0:
      if len(episode_len_) > 1:
        episode_len_ = np.insert(episode_len_, 0, 0)
        episode_len_ = episode_len_[1:] - episode_len_[:-1]
        writer.add_histogram(f'sim/{prefix}/sub_episode_len', episode_len_, env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_min', episode_len_[1:].min(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_max', episode_len_[1:].max(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_mean', episode_len_[1:].mean(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_std', episode_len_[1:].std(), env_step)

  writer.flush()

def make_env(cfg, writer, prefix, datadir, store, seed=0):

  suite, task = cfg.env.name.split('_', 1)

  if suite == 'atari':
    env = Atari(
        task, cfg.env.action_repeat, (64, 64), grayscale=cfg.env.grayscale,
        life_done=False, sticky_actions=True, seed=seed, all_actions=cfg.env.all_actions)
    env = OneHotAction(env)

  elif suite == 'crafter':
    env = Crafter(task, (64, 64), seed)
    env = OneHotAction(env)

  else:
    raise NotImplementedError(suite)

  env = TimeLimit(env, cfg.env.time_limit, cfg.env.time_penalty)

  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(
      lambda ep: summarize_episode(ep, cfg, datadir, writer, prefix))
  env = Collect(env, callbacks, cfg.env.precision)
  env = RewardObs(env)

  return env
