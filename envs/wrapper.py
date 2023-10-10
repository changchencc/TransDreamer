import numpy as np
from mlagents_envs.environment import UnityEnvironment
from pyvirtualdisplay import Display
import os
import pdb
from multiprocessing import Pool, Pipe, Process
import cloudpickle
import pickle
from torch import distributions as torchd

class CloudpickleWrapper(object):
  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    self.x = pickle.loads(ob)

  def __call__(self):
    return self.x()
def worker(remote, parent_remote, env_fn):
  parent_remote.close()
  env = env_fn()
  while True:
    cmd, data = remote.recv()

    if cmd == 'step':
      remote.send(env.step())

    elif cmd == 'reset':
      remote.send(env.reset())

    elif cmd == 'get_steps':
      decision, terminal = env.get_steps('BananaAgent?team=0')
      state = np.transpose(decision.obs[0], (0,3,1,2))
      # state_1 = decision.obs[0].reshape(8, 35)
      # state_2 = decision.obs[1].reshape(8, 2)
      # state = np.concatenate([state_1, state_2], axis=1)

      remote.send(state)

    elif cmd == 'set_actions':
      env.set_actions('BananaAgent?team=0', data)
      env.step()

      decision, terminal = env.get_steps('BananaAgent?team=0')
      if terminal.obs[0].shape[0] != 0:
        done = np.array([1]*4)
      else:
        done = np.array([0]*4)

      reward = decision.reward
      # state_1 = decision.obs[0].reshape(8, 35)
      # state_2 = decision.obs[1].reshape(8, 2)
      # next_state = np.concatenate([state_1, state_2], axis=1)
      next_state = np.transpose(decision.obs[0], (0,3,1,2))

      remote.send((next_state, reward, done))

    elif cmd == 'close':
      env.close()
      remote.close()
      break

    else:
      raise NotImplentedError

class SubprocVecEnv():
  def __init__(self, env_fns, agent_num):
    self.waiting = False
    self.closed = False
    no_of_envs = len(env_fns)
    self.remotes, self.work_remotes = \
      zip(*[Pipe() for _ in range(no_of_envs)])
    self.ps = []
    self.agent_num = agent_num

    for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
      proc = Process(target = worker,
        args = (wrk, rem, CloudpickleWrapper(fn)))
      self.ps.append(proc)

    for p in self.ps:
      p.daemon = True
      p.start()

    for remote in self.work_remotes:
      remote.close()

  def step_async(self, actions):
    if self.waiting:
      raise AlreadySteppingError
    self.waiting = True

    for remote, action in zip(self.remotes, actions):
      remote.send(('set_actions', action.reshape(self.agent_num, 1)))

  def step_wait(self):
    if not self.waiting:
      raise NotSteppingError
    self.waiting = False

    results = [remote.recv() for remote in self.remotes]
    next_states, rewards, dones = zip(*results)
    return np.concatenate(next_states, axis=0), np.concatenate(rewards, axis=0), np.concatenate(dones, axis=0)

  def set_actions(self, actions):
    self.step_async(actions)
    return self.step_wait()

  def get_steps(self):
    for remote in self.remotes:
      remote.send(('get_steps', None))

    states = [remote.recv() for remote in self.remotes]
    return np.concatenate(states, axis=0)

  def reset(self):
    for remote in self.remotes:
      remote.send(('reset', None))

    return np.stack([remote.recv() for remote in self.remotes])

  def close(self):
    if self.closed:
      return
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()
    self.closed = True

def make_mp_envs(env_file, num_env, agent_num):
  def make_env(env_file, worker_id):
    def fn():
      env = UnityEnvironment(file_name=env_file, worker_id=worker_id, timeout_wait=120)
      return env
    return fn
  return SubprocVecEnv([make_env(env_file, i) for i in range(num_env)], agent_num)

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
    return self.random_actor.sample()[0]