import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
import pdb

class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=True, seed=0):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
      env.seed(seed=seed)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale
    self._size = size
    self.action_size = 18

  @property
  def observation_space(self):
    shape = (1 if self._grayscale else 3,) + self._size
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})
    # return gym.spaces.Dict({
    #     'image': self._env.observation_space,
    # })

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    image = np.transpose(image, (2, 0, 1)) # 3, 64, 64
    return {'image': image}

  def step(self, action):
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    image = np.transpose(image, (2, 0, 1)) # 3, 64, 64
    obs = {'image': image}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)

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
