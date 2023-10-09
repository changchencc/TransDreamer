import atexit
import threading

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import cv2
import torch
from torch import distributions as torchd
import numpy as np
import pdb

class GymGridEnv():
  LOCK = threading.Lock()

  def __init__(self, name, action_repeat, max_steps=245, life_done=False):

    with self.LOCK:
      env = gym.make(name)
      env = RGBImgPartialObsWrapper(env, tile_size=9)  # Get pixel observations, (63, 63, 3)
      self._env = ImgObsWrapper(env)  # Get rid of the 'mission' field
      self._env.max_steps = max_steps
    self.action_repeat = action_repeat
    self._step_counter = 0
    self._random = np.random.RandomState(seed=None)
    self.life_done = life_done
    self.max_steps = max_steps
    self.action_size = 6

  def reset(self):

    self._step_counter = 0  # Reset internal timer
    with self.LOCK:
      observation = self._env.reset()

    # observation = self._env.render(mode='rgb_array')
    observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
    observation = np.clip(observation, 0, 255).astype(np.uint8)
    observation = np.transpose(observation, (2, 0, 1)) # 3, 64, 64
    self._step_counter = 0

    return {'image': observation}

  def step(self, action):
    reward = 0
    RESET = False
    for k in range(self.action_repeat):
      observation, reward_k, done, info = self._env.step(action)
      reward += reward_k
      self._step_counter += 1  # Increment internal timer

      if done:
        observation = self._env.reset()
        RESET = True

      if self.life_done:
        done = self._step_counter == self.max_steps

      if RESET:
        break

    # observation = self._env.render(mode='rgb_array')
    observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
    observation = np.clip(observation, 0, 255).astype(np.uint8)
    observation = np.transpose(observation, (2, 0, 1)) # 3, 64, 64

    return {'image': observation}, reward, done, info

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_space(self):
    shape = (3, 64, 64)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(self.action_size)

class OneHotAction():
  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def sample_random_action(self):
    action = np.zeros((1, self._env.action_space.n,), dtype=np.float)
    idx = np.random.randint(0, self._env.action_space.n, size=(1,))[0]
    action[0, idx] = 1
    return action

class TimeLimit():
  def __init__(self, env, duration, time_penalty):
    self._env = env
    self._step = None
    self._duration = duration
    self.time_penalty = time_penalty

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self.time_penalty:
      reward = reward - 1. / self._duration

    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()

class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    action = action[0]
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    transition['done'] = float(done)
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    obs['image'] = obs['image'][None,...]
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.n)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    transition['done'] = 0.0
    self._episode = [transition]
    obs['image'] = obs['image'][None,...]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      pdb.set_trace()
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)

class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs

class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
      np.isfinite(env.action_space.low),
      np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

    self.random_actor = torchd.independent.Independent(
      torchd.uniform.Uniform(torch.Tensor(env.action_space.low)[None],
                             torch.Tensor(env.action_space.high)[None]), 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)

  def sample_random_action(self):
    return self.random_actor.sample()[0].numpy()
