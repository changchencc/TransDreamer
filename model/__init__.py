import torch
from .transdreamer import TransDreamer

def get_model(cfg, device, seed=0):

  torch.manual_seed(seed=seed)

  if cfg.model == 'dreamer_transformer':
    model = TransDreamer(cfg)

  return model

