import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import get_dataloader
from utils import Checkpointer
from solver import get_optimizer
from vis.vis_logger import log_vis
import os
import numpy as np
import pdb

def vis(model, cfg):

  train_dl = get_dataloader(cfg.data.dataset, 'train', cfg.train.batch_size, cfg.data.data_root)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = model.to(device)
  optimizer = get_optimizer(cfg, model)

  checkpointer = Checkpointer(os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name), max_num=cfg.checkpoint.max_num)

  if cfg.resume:
    checkpoint = checkpointer.load(cfg.resume_ckpt)

    if checkpoint:

      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      global_step = checkpoint['global_step']

    else:
      start_epoch = 0
      global_step = 0

  else:

    start_epoch = 0
    global_step = 0

  writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.exp_name))
  total_iter = cfg.train.epoch * len(train_dl)

  seq_T = cfg.arch.seq_T
  T = seq_T[sum(global_step > np.array(cfg.train.curriculum))]

  for e in range(start_epoch, cfg.train.epoch):
    for i, (imgs, trj) in enumerate(train_dl):

      imgs = imgs.to(device)
      trj = trj.to(device)

      loss, loss_log, vis_log = model(imgs, trj, T)

      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.)
      optimizer.step()

      if global_step % cfg.train.print_every_step == 0:
        print('================Training {} / {}, Seq_T: {}================'.format(global_step+1, total_iter, T))
        for k, v in loss_log.items():
          print('{}: {:.6f}\n'.format(k, v))

      if global_step % cfg.train.log_every_step == 0:
        for k, v in loss_log.items():
          writer.add_scalar(k, v, global_step + 1)
        for k, v, in vis_log.items():
          writer.add_histogram('act/'+k, v, global_step+1)

        log_vis(writer, vis_log, imgs[:, :cfg.train.max_seq_T], global_step+1, cfg)

      if global_step % cfg.train.checkpoint_every_step == 0:
        checkpointer.save('', model, optimizer, global_step, e)

      global_step += 1

