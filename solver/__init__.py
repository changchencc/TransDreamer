import pdb

from torch import optim

# def get_parameters(modules):
#   """
#   Given a list of torch modules, returns a list of their parameters.
#   :param modules: iterable of modules
#   :returns: a list of parameters
#   """
#   model_parameters = []
#   for module in modules:
#     model_parameters += list(module.parameters())
#   return model_parameters

def get_parameters(modules):
  """
  Given a list of torch modules, returns a list of their parameters.
  :param modules: iterable of modules
  :returns: a list of parameters
  """

  model_parameters = []
  assert isinstance(modules, dict), 'only support dictionary in get_params.'
  for k, module in modules.items():
    model_parameters += list(module.parameters())

  return model_parameters

def get_optimizer(cfg, model, params=None):

  if cfg.optimize.optimizer == 'adam':
    opt_fn = optim.Adam
  if cfg.optimize.optimizer == 'adamW':
    opt_fn = optim.AdamW

  kwargs = {
    'weight_decay': cfg.optimize.weight_decay,
    'eps': cfg.optimize.eps
  }

  if cfg.arch.world_model.transformer.warm_up:
    transformer_named_param = model.world_model.dynamic.cell.named_parameters()
    transformer_param = [p for n, p in transformer_named_param]
    model_named_param = model.world_model.named_parameters()
    model_param = [p for n, p in model_named_param if 'dynamic.cell' not in n]

    model_optimizer = opt_fn(model_param, lr=cfg.optimize.model_lr, **kwargs)
    transformer_optimizer = opt_fn(transformer_param, lr=cfg.optimize.model_lr, **kwargs)
  else:
    model_optimizer = opt_fn(model.world_model.parameters(), lr=cfg.optimize.model_lr, **kwargs)
    transformer_optimizer = None

  actor_optimizer = opt_fn(model.actor.parameters(), lr=cfg.optimize.actor_lr, **kwargs)

  value_optimizer = opt_fn(model.value.parameters(), lr=cfg.optimize.value_lr, **kwargs)

  return {
    'model_optimizer': model_optimizer,
    'transformer_optimizer': transformer_optimizer,
    'value_optimizer': value_optimizer,
    'actor_optimizer': actor_optimizer,
  }