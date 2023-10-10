from torch.distributions.bernoulli import Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Independent, Normal, Bernoulli
from .utils import Conv2DBlock, ConvTranspose2DBlock, Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, RelaxedOneHotCategorical
from .distributions import SafeTruncatedNormal, ContDist
from .utils import Conv2DBlock, ConvTranspose2DBlock, Linear, MLP, GRUCell, LayerNormGRUCell, LayerNormGRUCellV2
from .transformer import Transformer
from collections import defaultdict
import numpy as np
import pdb
import time

class TransformerWorldModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.dynamic = TransformerDynamic(cfg)

    self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
    self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
    self.discrete_type = cfg.arch.world_model.discrete_type
    d_model = cfg.arch.world_model.transformer.d_model
    self.d_model = d_model
    deter_type = cfg.arch.world_model.transformer.deter_type
    n_layers = cfg.arch.world_model.transformer.n_layers
    if deter_type == 'concat_o':
      d_model = n_layers * d_model

    if self.stoch_discrete:
      dense_input_size = d_model + self.stoch_size * self.stoch_discrete
    else:
      dense_input_size = d_model + self.stoch_size

    self.img_dec = ImgDecoder(cfg, dense_input_size)
    self.reward = DenseDecoder(dense_input_size, cfg.arch.world_model.reward.layers, cfg.arch.world_model.reward.num_units, (1,),
                               act=cfg.arch.world_model.reward.act)

    self.pcont = DenseDecoder(dense_input_size, cfg.arch.world_model.pcont.layers, cfg.arch.world_model.pcont.num_units, (1,),
                              dist='binary', act='elu')

    self.r_transform = dict(
      tanh = torch.tanh,
      sigmoid = torch.sigmoid,
      none=nn.Identity(),
    )[cfg.rl.r_transform]

    self.discount = cfg.rl.discount
    self.lambda_ = cfg.rl.lambda_

    self.reward_layer = cfg.arch.world_model.reward_layer
    self.pcont_scale = cfg.loss.pcont_scale
    self.kl_scale = cfg.loss.kl_scale
    self.kl_balance = cfg.loss.kl_balance
    self.free_nats = cfg.loss.free_nats
    self.H = cfg.arch.H
    self.grad_clip = cfg.optimize.grad_clip
    self.action_size = cfg.env.action_size
    self.log_every_step = cfg.train.log_every_step
    self.batch_length = cfg.train.batch_length
    self.grayscale = cfg.env.grayscale
    self.slow_update = 0
    self.n_sample = cfg.train.n_sample
    self.imag_last_T = cfg.train.imag_last_T
    self.slow_update_step = cfg.slow_update_step
    self.log_grad = cfg.train.log_grad
    self.input_type = cfg.arch.world_model.input_type

  def forward(self, traj):
    raise NotImplementedError

  def compute_loss(self, traj, global_step, temp):

    self.train()
    self.requires_grad_(True)

    # world model rollout to obtain state representation
    prior_state, post_state = self.dynamic(traj, None, temp)

    # compute world model loss given state representation
    model_loss, model_logs = self.world_model_loss(global_step, traj, prior_state, post_state, temp)

    return model_loss, model_logs, prior_state, post_state

  def world_model_loss(self, global_step, traj, prior_state, post_state, temp):

    obs = traj[self.input_type]
    obs = obs / 255. - 0.5
    reward = traj['reward']
    reward = self.r_transform(reward).float()

    post_state_trimed = {}
    for k, v in post_state.items():
      if k in ['stoch', 'logits', 'mean', 'std']:
        post_state_trimed[k] = v[:, 1:]
      else:
        post_state_trimed[k] = v

    rnn_feature = self.dynamic.get_feature(post_state_trimed, layer=self.reward_layer)
    seq_len = self.H

    image_pred_pdf = self.img_dec(rnn_feature)  # B, T-1, 3, 64, 64
    reward_pred_pdf = self.reward(rnn_feature)  # B, T-1, 1

    pred_pcont = self.pcont(rnn_feature)  # B, T, 1
    pcont_target = self.discount * (1. - traj['done'][:, 1:].float())  # B, T
    pcont_loss = -(pred_pcont.log_prob((pcont_target.unsqueeze(2) > 0.5).float())).sum(-1) / seq_len #
    pcont_loss = self.pcont_scale * pcont_loss.mean()
    discount_acc = ((pred_pcont.mean == pcont_target.unsqueeze(2)).float().squeeze(-1)).sum(-1) / seq_len
    discount_acc = discount_acc.mean()

    image_pred_loss = -(image_pred_pdf.log_prob(obs[:, 1:])).sum(-1).float() / seq_len  # B
    image_pred_loss = image_pred_loss.mean()
    mse_loss = (F.mse_loss(image_pred_pdf.mean, obs[:, 1:], reduction='none').flatten(start_dim=-3).sum(-1)).sum(-1) / seq_len
    mse_loss = mse_loss.mean()
    reward_pred_loss = -(reward_pred_pdf.log_prob(reward[:, 1:].unsqueeze(2))).sum(-1) / seq_len # B
    reward_pred_loss = reward_pred_loss.mean()
    pred_reward = reward_pred_pdf.mean

    prior_dist = self.dynamic.get_dist(prior_state, temp)
    post_dist = self.dynamic.get_dist(post_state_trimed, temp)

    value_lhs = kl_divergence(post_dist, self.dynamic.get_dist(prior_state, temp, detach=True)) # B, T
    value_rhs = kl_divergence(self.dynamic.get_dist(post_state_trimed, temp, detach=True), prior_dist)
    value_lhs = value_lhs.sum(-1) / seq_len
    value_rhs = value_rhs.sum(-1) / seq_len
    loss_lhs = torch.maximum(value_lhs.mean(), value_lhs.new_ones(value_lhs.mean().shape) * self.free_nats)
    loss_rhs = torch.maximum(value_rhs.mean(), value_rhs.new_ones(value_rhs.mean().shape) * self.free_nats)

    kl_loss = (1. - self.kl_balance) * loss_lhs + self.kl_balance * loss_rhs
    kl_scale = self.kl_scale
    kl_loss = kl_scale * kl_loss

    model_loss = image_pred_loss + reward_pred_loss + kl_loss + pcont_loss

    if global_step % self.log_every_step == 0:
      post_dist = Independent(OneHotCategorical(logits=post_state_trimed['logits']), 1)
      prior_dist = Independent(OneHotCategorical(logits=prior_state['logits']), 1)
      logs = {
        'model_loss': model_loss.detach().item(),
        'model_kl_loss': kl_loss.detach().item(),
        'model_reward_logprob_loss': reward_pred_loss.detach().item(),
        'model_image_loss': image_pred_loss.detach().item(),
        'model_mse_loss': mse_loss.detach(),
        'ACT_prior_state': {k: v.detach() for k, v in prior_state.items()},
        'ACT_prior_entropy': prior_dist.entropy().mean().detach().item(),
        'ACT_post_state': {k: v.detach() for k, v in post_state.items()},
        'ACT_post_entropy': post_dist.entropy().mean().detach().item(),
        'ACT_gt_reward': reward[:, 1:],
        'dec_img': (image_pred_pdf.mean.detach() + 0.5),  # B, T, 3, 64, 64
        'gt_img': obs[:, 1:] + 0.5,
        'reward_input': rnn_feature.detach(),
        'model_discount_logprob_loss': pcont_loss.detach().item(),
        'discount_acc': discount_acc.detach(),
        'pred_reward': pred_reward.detach().squeeze(-1),
        'pred_discount': pred_pcont.mean.detach().squeeze(-1),
        'hp_kl_scale': kl_scale,
      }

    else:
      logs = {}

    return model_loss, logs

  def imagine_ahead(self, actor, post_state, traj, sample_len, temp):
    """
    post_state:
      mean: mean of q(s_t | h_t, o_t), (B*T, H)
      std: std of q(s_t | h_t, o_t), (B*T, H)
      stoch: s_t sampled from q(s_t | h_t, o_t), (B*T, H)
      deter: h_t, (B*T, H)
    """

    self.eval()
    self.requires_grad_(False)

    action = traj['action']

    # randomly choose a state to start imagination
    min_idx = self.H - 2 # trimed the last step, at least imagine 2 steps for TD target
    perm = torch.randperm(min_idx, device=action.device)
    min_idx = perm[0] + 1

    pred_state = defaultdict(list)

    # pred_prior = {k: v.detach()[:, :min_idx] for k, v in post_state_trimed.items()}
    post_stoch = post_state['stoch'][:, :min_idx]
    action = action[:, 1: min_idx+1]
    imag_rnn_feat_list = []
    imag_action_list = []

    for t in range(self.batch_length - min_idx):

      pred_prior = self.dynamic.infer_prior_stoch(post_stoch[:, -sample_len:], temp, action[:, -sample_len:])
      rnn_feature = self.dynamic.get_feature(pred_prior, layer=self.reward_layer)

      pred_action_pdf = actor(rnn_feature[:, -1:].detach())
      imag_action = pred_action_pdf.sample()
      imag_action = imag_action + pred_action_pdf.mean - pred_action_pdf.mean.detach()  # straight through
      action = torch.cat([action, imag_action], dim=1)

      for k, v in pred_prior.items():
        pred_state[k].append(v[:, -1:])
      post_stoch = torch.cat([post_stoch, pred_prior['stoch'][:, -1:]], dim=1)

      imag_rnn_feat_list.append(rnn_feature[:, -1:])
      imag_action_list.append(imag_action)

    for k, v in pred_state.items():
      pred_state[k] = torch.cat(v, dim=1)
    actions = torch.cat(imag_action_list, dim=1)
    rnn_features = torch.cat(imag_rnn_feat_list, dim=1)

    reward = self.reward(rnn_features).mean
    discount = self.discount * self.pcont(rnn_features).mean

    return rnn_features, pred_state, actions, reward, discount, min_idx

  def optimize_world_model(self, model_loss, model_optimizer, transformer_optimizer, writer, global_step):

    model_loss.backward()
    grad_norm_model = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
    if (global_step % self.log_every_step == 0) and self.log_grad:
      for n, p in self.named_parameters():
        if p.requires_grad:
          try:
            writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
          except:
            pdb.set_trace()
    model_optimizer.step()
    if transformer_optimizer is not None:
      transformer_optimizer.step()

    return grad_norm_model.item()


class TransformerDynamic(nn.Module):

  def __init__(self, cfg):
    super().__init__()

    act = cfg.arch.world_model.RSSM.act
    hidden_size = cfg.arch.world_model.RSSM.hidden_size
    action_size = cfg.env.action_size
    self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
    self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
    self.discrete_type = cfg.arch.world_model.discrete_type
    self.ST = cfg.arch.world_model.RSSM.ST
    self.d_model = cfg.arch.world_model.transformer.d_model
    self.pre_lnorm = cfg.arch.world_model.transformer.pre_lnorm
    self.act_after_emb = cfg.arch.world_model.act_after_emb

    self.img_enc = ImgEncoder(cfg)

    weight_init = cfg.arch.world_model.RSSM.weight_init
    self.cell = Transformer(cfg.arch.world_model.transformer)

    if self.stoch_discrete:
      latent_dim = self.stoch_size*self.stoch_discrete
      latent_dim_out = latent_dim
    else:
      latent_dim = self.stoch_size
      latent_dim_out = latent_dim * 2
    self.act_stoch_mlp = Linear(action_size + latent_dim, self.d_model, weight_init=weight_init)
    self.q_trans = cfg.arch.q_trans
    self.q_emb_action = cfg.arch.world_model.q_emb_action
    q_emb_size = 1536
    if self.q_emb_action:
      q_emb_size = q_emb_size + action_size

    if self.q_trans:
      self.q_trans_model = Transformer(cfg.arch.world_model.q_transformer)
      self.q_emb = nn.Sequential(
        Linear(q_emb_size, self.d_model),
        nn.ELU())

      self.q_trans_deter_type = cfg.arch.world_model.q_transformer.deter_type
      self.q_trans_layers = cfg.arch.world_model.q_transformer.n_layers
      if self.q_trans_deter_type == 'concat_o':
        d_model = self.q_trans_layers * self.d_model
      else:
        d_model = self.d_model
      self.post_stoch_mlp = MLP([d_model, hidden_size, latent_dim_out], act=act,
                                weight_init=weight_init)

    else:
      self.post_stoch_mlp = MLP([q_emb_size, hidden_size, latent_dim_out], act=act,
                                weight_init=weight_init)

    self.deter_type = cfg.arch.world_model.transformer.deter_type
    self.n_layers = cfg.arch.world_model.transformer.n_layers
    if self.deter_type == 'concat_o':
      d_model = self.n_layers * self.d_model
    else:
      d_model = self.d_model
    self.prior_stoch_mlp = MLP([d_model, hidden_size, latent_dim_out], act=act, weight_init=weight_init)
    self.input_type = cfg.arch.world_model.input_type

  def forward(self, traj, prev_state, temp):
    """
    traj:
      observations: embedding of observed images, B, T, C
      actions: (one-hot) vector in action space, B, T, d_act
      dones: scalar, B, T

    prev_state:
      deter: GRU hidden state, B, h1
      stoch: RSSM stochastic state, B, h2
    """

    obs = traj[self.input_type]
    obs = obs / 255. - 0.5
    obs_emb = self.img_enc(obs) # B, T, C

    actions = traj['action']
    dones = traj['done']

    # q(s_t | o_t)
    post = self.infer_post_stoch(obs_emb, temp, action=None)
    s_t = post['stoch'][:, :-1]
    # p(s_(t+1) | s_t, a_t)
    prior = self.infer_prior_stoch(s_t, temp, actions[:, 1:])

    post['deter'] = prior['deter']
    post['o_t'] = prior['o_t']

    if self.stoch_discrete:
      prior['stoch_int'] = prior['stoch'].argmax(-1).float()
      post['stoch_int'] = post['stoch'].argmax(-1).float()

    return prior, post

  def get_feature(self, state, layer=None):

    if self.stoch_discrete:
      shape = state['stoch'].shape
      stoch = state['stoch'].reshape([*shape[:-2]] + [self.stoch_size*self.stoch_discrete])

      if layer:
        o_t = state['o_t']  # B, T, L, D
        deter = o_t[:, :, layer]
      else:
        # assert self.deter_type == 'concat_o'
        deter = state['deter']

      return torch.cat([stoch, deter], dim=-1)  # B, T, 2H

    else:
      stoch = state['stoch']

      if layer:
        o_t = state['o_t']  # B, T, L, D
        deter = o_t[:, :, layer]
      else:
        assert self.deter_type == 'concat_o'
        deter = state['deter']

      return torch.cat([stoch, deter], dim=-1)  # B, T, 2H


  def get_dist(self, state, temp, detach=False):
    if self.stoch_discrete:
      return self.get_discrete_dist(state, temp, detach)
    else:
      return self.get_normal_dist(state, detach)

  def get_normal_dist(self, state, detach):

    mean = state['mean']
    std = state['std']

    if detach:
      mean = mean.detach()
      std = std.detach()

    return Independent(Normal(mean, std), 1)

  def get_discrete_dist(self, state, temp, detach):

    logits = state['logits']

    if detach:
      logits = logits.detach()

    if self.discrete_type == 'discrete':
      dist = Independent(OneHotCategorical(logits=logits), 1)

    if self.discrete_type == 'gumbel':
      try:
        dist = Independent(MyRelaxedOneHotCategorical(temp, logits=logits), 1)
      except:
        pdb.set_trace()

    return dist

  def encode_s(self, prev_stoch, action):

    B, T, N, C = prev_stoch.shape
    prev_stoch = prev_stoch.reshape(B, T, N * C)
    act_sto_emb = self.act_stoch_mlp(torch.cat([action, prev_stoch], dim=-1))

    act_sto_emb = F.elu(act_sto_emb)

    return act_sto_emb

  def infer_prior_stoch(self, prev_stoch, temp, actions):

    B, T = prev_stoch.shape[:2]
    if self.stoch_discrete:
      B, T, N, C = prev_stoch.shape

      act_sto_emb = self.encode_s(prev_stoch, actions)
    else:
      act_sto_emb = self.act_stoch_mlp(torch.cat([prev_stoch, actions], dim=-1))
      act_sto_emb = F.elu(act_sto_emb)

    s_t_reshape = act_sto_emb.reshape(B, T, -1, 1, 1)
    o_t = self.cell(s_t_reshape, None) # B, T, L, D, H, W

    o_t = o_t.reshape(B, T, self.n_layers, -1)
    if self.deter_type == 'concat_o':
      deter = o_t.reshape(B, T, -1)
    else:
      deter = o_t[:, :, -1]
    pred_logits = self.prior_stoch_mlp(deter).float()

    if self.stoch_discrete:
      B, T, N, C = prev_stoch.shape
      pred_logits = pred_logits.reshape(B, T, N, C)

    prior_state = self.stat_layer(pred_logits, temp)
    prior_state.update({
      'deter': deter,
      'o_t': o_t,
    })

    return prior_state

  def infer_post_stoch(self, observation, temp, action=None):

    if action is not None:
      observation = torch.cat([observation, action], dim=-1)
    B, T, C = observation.shape
    if self.q_trans:
      q_emb = self.q_emb(observation)
      e = self.q_trans_model(q_emb.reshape(B, T, self.d_model, 1, 1), None)

      e = e.reshape(B, T, self.q_trans_layers, -1)
      if self.q_trans_deter_type == 'concat_o':
        e = e.reshape(B, T, -1)
      else:
        e = e[:, :, -1]
      logits = self.post_stoch_mlp(e).float()

    else:
      logits = self.post_stoch_mlp(observation).float()

    if self.stoch_discrete:
      logits = logits.reshape(B, T, self.stoch_discrete, self.stoch_size).float()
    post_state = self.stat_layer(logits, temp)

    return post_state

  def stat_layer(self, logits, temp):

    if self.stoch_discrete:

      if self.discrete_type == 'discrete':
        # print(f'logits min: {logits.min()}')
        # print(f'logits mean: {logits.mean()}')
        # print(f'logits max: {logits.max()}')
        dist = Independent(OneHotCategorical(logits=logits), 1)
        stoch = dist.sample()
        stoch = stoch + dist.mean - dist.mean.detach()

      if self.discrete_type == 'gumbel':
        try:
          dist = Independent(MyRelaxedOneHotCategorical(temp, logits=logits), 1)
        except:
          pdb.set_trace()
        stoch = dist.rsample()

      state = {
        'logits': logits,
        'stoch': stoch,
      }

    else:

      mean, std = logits.float().chunk(2, dim=-1)
      std = 2. * (std / 2.).sigmoid()
      dist = Normal(mean, std)
      stoch = dist.rsample()

      state = {
        'mean': mean,
        'std': std,
        'stoch': stoch,
      }

    return state

class ImgEncoder(nn.Module):

  def __init__(self, cfg):
    super().__init__()

    self.q_trans = cfg.arch.q_trans
    depth = 48
    c_in = 1 if cfg.env.grayscale else 3
    self.enc = nn.Sequential(
      Conv2DBlock(c_in, depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
      Conv2DBlock(depth, 2*depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
      Conv2DBlock(2*depth, 4*depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
      Conv2DBlock(4*depth, 8*depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=not self.q_trans, act='elu', weight_init='xavier'),
    )

  def forward(self, ipts):
    """
    ipts: tensor, (B, T, 3, 64, 64)
    return: tensor, (B, T, 1024)
    """

    shapes = ipts.shape
    o = self.enc(ipts.view([-1] + [*shapes[-3:]]))
    o = o.reshape([*shapes[:-3]] + [1536])

    return o

class ImgDecoder(nn.Module):

  def __init__(self, cfg, input_size):
    super().__init__()

    depth = 48
    self.c_out = 1 if cfg.env.grayscale else 3
    self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete

    self.fc = Linear(input_size, 1536, bias=True, weight_init='xavier')
    if cfg.arch.decoder.dec_type == 'conv':
      self.dec = nn.Sequential(
        ConvTranspose2DBlock(1536, 4*depth, 5, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
        ConvTranspose2DBlock(4*depth, 2*depth, 5, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
        ConvTranspose2DBlock(2*depth, depth, 6, 2, 0, num_groups=0, bias=True, non_linearity=True, act='elu', weight_init='xavier'),
        ConvTranspose2DBlock(depth, self.c_out, 6, 2, 0, num_groups=0, bias=True, non_linearity=False, weight_init='xavier'),
      )

    elif cfg.dec_type == 'pixelshuffle':
      pass

    else:
      raise ValueError(f"decoder type {cfg.dec_type} is not supported.")

    self.shape = (self.c_out, 64, 64)
    self.rec_sigma = cfg.arch.world_model.rec_sigma


  def forward(self, ipts):
    """
    ipts: tensor, (B, T, C)
    """

    shape = ipts.shape

    fc_o = self.fc(ipts)
    dec_o = self.dec(fc_o.reshape(shape[0]*shape[1], 1536, 1, 1))
    dec_o = dec_o.reshape([*shape[:2]] + [self.c_out, 64, 64] )

    dec_pdf = Independent(Normal(dec_o, self.rec_sigma * dec_o.new_ones(dec_o.shape)), len(self.shape))

    return dec_pdf

class DenseDecoder(nn.Module):

  def __init__(self, input_size, layers, units, output_shape, weight_init='xavier', dist='normal', act='relu'):
    super().__init__()

    acts = {
      'relu': nn.ReLU,
      'elu': nn.ELU,
      'celu': nn.CELU,
    }
    module_list = []

    for i in range(layers):

      if i == 0:
        dim_in = input_size
      else:
        dim_in = units
      dim_out = units

      module_list.append(Linear(dim_in, dim_out, weight_init=weight_init))
      module_list.append(acts[act]())

    module_list.append(Linear(dim_out, 1, weight_init=weight_init))
    self.dec = nn.Sequential(*module_list)

    self.dist = dist
    self.output_shape = output_shape

  def forward(self, inpts):

    logits = self.dec(inpts)
    logits = logits.float()

    if self.dist == 'normal':
      pdf = Independent(Normal(logits, 1), len(self.output_shape))


    elif self.dist == 'binary':
      pdf = Independent(Bernoulli(logits=logits), len(self.output_shape))

    else:
      raise NotImplementedError(self.dist)

    return pdf

class ActionDecoder(nn.Module):

  def __init__(self, input_size, action_size, layers, units, dist='onehot', act='relu',
                min_std=0.1, init_std=5, mean_scale=5, weight_init='xavier'):
    super().__init__()

    acts = {
      'relu': nn.ReLU,
      'elu': nn.ELU,
      'celu': nn.CELU,
    }
    module_list = []

    for i in range(layers):

      if i == 0:
        dim_in = input_size
      else:
        dim_in = units
      dim_out = units

      module_list.append(Linear(dim_in, dim_out, weight_init=weight_init))
      module_list.append(acts[act]())

    if dist == 'trunc_normal':
      module_list.append(Linear(dim_out, 2*action_size, weight_init=weight_init))

    elif dist == 'onehot':
      module_list.append(Linear(dim_out, action_size, weight_init=weight_init))

    else:
      raise NotImplementedError(self.dist)

    self.dec = nn.Sequential(*module_list)
    self.dist = dist
    self.raw_init_std = np.log(np.exp(init_std) - 1)
    self.min_std = min_std
    self.mean_scale = mean_scale

  def forward(self, inpts):

    logits = self.dec(inpts)

    logits = logits.float()

    if self.dist == 'trunc_normal':

      mean, std = torch.chunk(logits, 2, -1)
      mean = torch.tanh(mean)
      std = 2 * torch.sigmoid(std / 2) + self.min_std
      dist = SafeTruncatedNormal(mean, std, -1, 1)
      dist = ContDist(Independent(dist, 1))

    if self.dist == 'onehot':

      dist = OneHotCategorical(logits=logits)

    return dist

class MyRelaxedOneHotCategorical(RelaxedOneHotCategorical):
  def __init__(self, temp, logits, eps=1e-16, validate_args=False):
    super(MyRelaxedOneHotCategorical, self).__init__(temp, logits=logits, validate_args=validate_args)
    """
    re-write the rsample() api.
    """
    self.dev = logits.device
    self.eps = eps

  def log_prob(self, value):
    K = self.logits.shape[-1]

    log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
                 self.temperature.log().mul(-(K - 1)))
    score = self.logits - value.log().mul(self.temperature)
    score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)

    log_prob = score + log_scale
    if torch.isinf(log_prob).any():
      pdb.set_trace()

    return log_prob

  def rsample(self, eps=1e-20):
    uniforms = torch.rand(self.logits.shape, dtype=self.logits.dtype, device=self.dev)
    uniforms = torch.clamp(uniforms, self.eps, 1. - self.eps)
    gumbels = - torch.log(- torch.log(uniforms))
    scores = (self.logits + gumbels) / self.temperature
    samples = (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()
    if torch.isnan(samples).any():
      pdb.set_trace()
    # if (1. - self.support.check(samples).float()).any():
    #   pdb.set_trace()
    return samples