import os
import pdb
import pickle
import torch.nn as nn
import torch
import numpy as np
# from .model.utils import spatial_transformer

class Checkpointer(object):
  def __init__(self, checkpoint_dir, max_num):
    self.max_num = max_num
    self.checkpoint_dir = checkpoint_dir

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.model_list_path = os.path.join(checkpoint_dir, 'model_list.pkl')

    if not os.path.exists(self.model_list_path):
      model_list = []
      with open(self.model_list_path, 'wb') as f:
        pickle.dump(model_list, f)

  def load(self, path, model_idx=1):
    if path == '':
      with open(self.model_list_path, 'rb') as f:
        model_list = pickle.load(f)

      if len(model_list) == 0:
        print('Start training from scratch.')
        return None

      else:
        n_files = len(model_list)
        model_idx = min(n_files, model_idx)
        path = model_list[-model_idx]
        print(f'Load checkpoint from {path}')

        checkpoint = torch.load(path)
        return checkpoint
    else:

      assert os.path.exists(path), f'checkpoint {path} not exits.'
      checkpoint = torch.load(path)

      return checkpoint

  def save(self, path, model, optimizers, global_step, env_step):

    if path == '':
      path = os.path.join(self.checkpoint_dir, 'model_{:09}.pth'.format(global_step + 1))

      with open(self.model_list_path, 'rb+') as f:
        model_list = pickle.load(f)
        if len(model_list) >= self.max_num:
          if os.path.exists(model_list[0]):
            os.remove(model_list[0])

          del model_list[0]
        model_list.append(path)
      with open(self.model_list_path, 'rb+') as f:
        pickle.dump(model_list, f)

    if isinstance(model, nn.DataParallel):
      model = model.module

    checkpoint = {
      'model': model.state_dict(),
      'global_step': global_step,
      'env_step': env_step,
    }
    if isinstance(optimizers, dict):
      for k, v in optimizers.items():
        if v is not None:
          checkpoint.update({
            k: v.state_dict(),
          })
    else:
      checkpoint.update({
        'optimizer': optimizers.state_dict(),
      })

    assert path.endswith('.pth')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
      torch.save(checkpoint, f)

    print(f'Saved checkpoint to {path}')
