import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from utils import Checkpointer
from solver import get_optimizer
from envs import make_env, count_steps
from data import EnvIterDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from pprint import pprint
import pdb
import torch.autograd.profiler as profiler
from time import time
from collections import defaultdict

def anneal_learning_rate(global_step, cfg):

  if (global_step - cfg.arch.prefill) < cfg.optimize.warmup_iter:
    # warmup
    lr = cfg.optimize.base_lr / cfg.optimize.warmup_iter * (global_step - cfg.arch.prefill)

  else:
    lr = cfg.optimize.base_lr

  # decay
  lr = lr * cfg.optimize.exp_rate ** ((global_step  - cfg.arch.prefill)/ cfg.optimize.decay_step)

  if (global_step - cfg.arch.prefill) > cfg.optimize.decay_step:
    lr = max(lr, cfg.optimize.end_lr)

  return lr

def anneal_temp(global_step, cfg):

  temp_start = cfg.arch.world_model.temp_start
  temp_end = cfg.arch.world_model.temp_end
  decay_steps = cfg.arch.world_model.temp_decay_steps
  temp = temp_start - (temp_start - temp_end) * (global_step - cfg.arch.prefill) / decay_steps

  temp = max(temp, temp_end)

  return temp

def simulate_test(model, test_env, cfg, global_step, device):

  model.eval()

  obs = test_env.reset()
  action_list = torch.zeros(1, 1, cfg.env.action_size).float()
  action_list[:, 0, 0] = 1. # B, T, C
  state = None
  done = False
  input_type = cfg.arch.world_model.input_type

  with torch.no_grad():
    while not done:
      next_obs, reward, done = test_env.step(action_list[0, -1].detach().cpu().numpy())
      prev_image = torch.tensor(obs[input_type])
      next_image = torch.tensor(next_obs[input_type])
      action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device), global_step, 0.1, state, training=False, context_len=cfg.train.batch_length)
      obs = next_obs

def train(model, cfg, device):

  print("======== Settings ========")
  pprint(cfg)

  print("======== Model ========")
  pprint(model)

  model = model.to(device)

  optimizers = get_optimizer(cfg, model)
  checkpointer_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
  checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)
  with open(checkpointer_path + '/config.yaml', 'w') as f:
    cfg.dump(stream=f, default_flow_style=False)
    print(f"config file saved to {checkpointer_path + '/config.yaml'}")

  if cfg.resume:
    checkpoint = checkpointer.load(cfg.resume_ckpt)

    if checkpoint:
      model.load_state_dict(checkpoint['model'])
      for k, v in optimizers.items():
        if v is not None:
          v.load_state_dict(checkpoint[k])
      env_step = checkpoint['env_step']
      global_step = checkpoint['global_step']

    else:
      env_step = 0
      global_step = 0

  else:
    env_step = 0
    global_step = 0

  writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)

  datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
  test_datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
  train_env = make_env(cfg, writer, 'train', datadir, store=True)
  test_env = make_env(cfg, writer, 'test', test_datadir, store=True)

  # fill in length of 5000 frames
  train_env.reset()
  steps = count_steps(datadir, cfg)
  length = 0
  while steps < cfg.arch.prefill:
    action = train_env.sample_random_action()
    next_obs, reward, done = train_env.step(action[0])
    length += 1
    steps += done * length
    length = length * (1. - done)
    if done:
      train_env.reset()

  steps = count_steps(datadir, cfg)
  print(f'collected {steps} steps. Start training...')
  train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
  train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
  train_iter = iter(train_dl)
  global_step = max(global_step, steps)

  obs = train_env.reset()
  state = None
  action_list = torch.zeros(1, 1, cfg.env.action_size).float() # T, C
  action_list[0, 0, 0] = 1.
  input_type = cfg.arch.world_model.input_type
  temp = cfg.arch.world_model.temp_start

  while global_step < cfg.total_steps:

    with torch.no_grad():
      model.eval()
      next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())
      prev_image = torch.tensor(obs[input_type])
      next_image = torch.tensor(next_obs[input_type])
      action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device),
                                        global_step, 0.1, state, context_len=cfg.train.batch_length)
      obs = next_obs
      if done:
        train_env.reset()
        state = None
        action_list = torch.zeros(1, 1, cfg.env.action_size).float()  # T, C
        action_list[0, 0, 0] = 1.

    if global_step % cfg.train.train_every == 0:

      temp = anneal_temp(global_step, cfg)

      model.train()

      traj = next(train_iter)
      for k, v in traj.items():
        traj[k] = v.to(device).float()

      logs = {}

      model_optimizer = optimizers['model_optimizer']
      model_optimizer.zero_grad()
      transformer_optimizer = optimizers['transformer_optimizer']
      if transformer_optimizer is not None:
        transformer_optimizer.zero_grad()
      model_loss, model_logs, prior_state, post_state = model.world_model_loss(global_step, traj, temp)
      grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, transformer_optimizer, writer, global_step)
      if cfg.arch.world_model.transformer.warm_up:
        lr = anneal_learning_rate(global_step, cfg)
        for param_group in transformer_optimizer.param_groups:
          param_group['lr'] = lr
      else:
        lr = cfg.optimize.model_lr

      actor_optimizer = optimizers['actor_optimizer']
      value_optimizer = optimizers['value_optimizer']
      actor_optimizer.zero_grad()
      value_optimizer.zero_grad()
      actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(global_step, post_state, traj, temp)
      grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
      grad_norm_value = model.optimize_value(value_loss, value_optimizer, writer, global_step)

      if global_step % cfg.train.log_every_step == 0:

        logs.update(model_logs)
        logs.update(actor_value_logs)
        model.write_logs(logs, traj, global_step, writer)

        writer.add_scalar('train_hp/lr', lr, global_step)

        grad_norm = dict(
          grad_norm_model = grad_norm_model,
          grad_norm_actor = grad_norm_actor,
          grad_norm_value = grad_norm_value,
        )

        for k, v in grad_norm.items():
          writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)

    # evaluate RL
    if global_step % cfg.train.eval_every_step == 0:
      simulate_test(model, test_env, cfg, global_step, device)

    if global_step % cfg.train.checkpoint_every_step == 0:
      env_step = count_steps(datadir, cfg)
      checkpointer.save('', model, optimizers, global_step, env_step)

    global_step += 1
