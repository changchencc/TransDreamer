from collections import defaultdict
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Independent
from .modules_transformer import TransformerWorldModel, DenseDecoder, ActionDecoder
import torch
import torch.nn as nn
import pdb

class TransDreamer(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.world_model = TransformerWorldModel(cfg)

    self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
    self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
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
    self.aggregator = cfg.arch.actor.aggregator
    if self.aggregator == 'attn':
      dense_input_size = dense_input_size + self.d_model
    self.actor = ActionDecoder(dense_input_size, cfg.env.action_size, cfg.arch.actor.layers, cfg.arch.actor.num_units,
                               dist=cfg.arch.actor.dist, init_std=cfg.arch.actor.init_std, act=cfg.arch.actor.act)

    self.value = DenseDecoder(dense_input_size, cfg.arch.value.layers, cfg.arch.value.num_units, (1,), act=cfg.arch.value.act)
    self.slow_value = DenseDecoder(dense_input_size, cfg.arch.value.layers, cfg.arch.value.num_units, (1,), act=cfg.arch.value.act)

    self.discount = cfg.rl.discount
    self.lambda_ = cfg.rl.lambda_

    self.actor_loss_type = cfg.arch.actor.actor_loss_type
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
    self.reward_layer = cfg.arch.world_model.reward_layer
    self.log_grad = cfg.train.log_grad
    self.ent_scale = cfg.loss.ent_scale
    self.action_dist = cfg.arch.actor.dist

    self.r_transform = dict(
      tanh = torch.tanh,
      sigmoid = torch.sigmoid,
      none = torch.nn.Identity(),
    )[cfg.rl.r_transform]

  def forward(self):
    raise NotImplementedError

  def write_logs(self, logs, traj, global_step, writer, tag='train', min_idx=None):

    rec_img = logs['dec_img']
    gt_img = logs['gt_img']  # B, {1:T}, C, H, W

    writer.add_video('train/rec - gt',
                      torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0., 1.).cpu(),
                      global_step=global_step)

    for k, v in logs.items():

      if 'loss' in k:
        writer.add_scalar(tag + '_loss/' + k, v, global_step=global_step)
      if 'grad_norm' in k:
        writer.add_scalar(tag + '_grad_norm/' + k, v, global_step=global_step)
      if 'hp' in k:
        writer.add_scalar(tag + '_hp/' + k, v, global_step=global_step)
      if 'ACT' in k:
        if isinstance(v, dict):
          for kk, vv in v.items():
            if isinstance(vv, torch.Tensor):
              writer.add_histogram(tag + '_ACT/' + k + '-' + kk, vv, global_step=global_step)
              writer.add_scalar(tag + '_mean_ACT/' + k + '-' + kk, vv.mean(), global_step=global_step)
            if isinstance(vv, float):
              writer.add_scalar(tag + '_ACT/' + k + '-'  + kk, vv, global_step=global_step)
        else:
          if isinstance(v, torch.Tensor):
            writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
            writer.add_scalar(tag + '_mean_ACT/' + k, v.mean(), global_step=global_step)
          if isinstance(v, float):
            writer.add_scalar(tag + '_ACT/' + k, v, global_step=global_step)
      if 'imag_value' in k:
        writer.add_scalar(tag + '_values/' + k, v.mean(), global_step=global_step)
        writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
      if 'actor_target' in k:
        writer.add_scalar(tag + 'actor_target/' + k, v, global_step=global_step)


  def optimize_actor(self, actor_loss, actor_optimizer, writer, global_step):

    actor_loss.backward()
    grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

    if (global_step % self.log_every_step == 0) and self.log_grad:
      for n, p in self.actor.named_parameters():
        if p.requires_grad:
          try:
            writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
          except:
            pdb.set_trace()

    actor_optimizer.step()

    return grad_norm_actor.item()

  def optimize_value(self, value_loss, value_optimizer,writer , global_step):

    value_loss.backward()
    grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)

    if (global_step % self.log_every_step == 0) and self.log_grad:
      for n, p in self.value.named_parameters():
        if p.requires_grad:
          try:
            writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
          except:
            pdb.set_trace()
    value_optimizer.step()

    return grad_norm_value.item()

  def world_model_loss(self, global_step, traj, temp):
    return self.world_model.compute_loss(traj, global_step, temp)

  def actor_and_value_loss(self, global_step, post_state, traj, temp):
    self.update_slow_target(global_step)
    self.value.eval()
    self.value.requires_grad_(False)

    imagine_feat, imagine_state, imagine_action, \
      imagine_reward, imagine_disc, imagine_idx = self.world_model.imagine_ahead(self.actor, post_state, traj, self.batch_length-1, temp)

    target, weights = self.compute_target(imagine_feat, imagine_reward, imagine_disc) # B*T, H-1, 1

    slice_idx = -1

    actor_dist = self.actor(imagine_feat.detach()) # B*T, H
    if self.action_dist == 'onehot':
      indices = imagine_action.max(-1)[1]
      actor_logprob = actor_dist._categorical.log_prob(indices)
    else:
      actor_logprob = actor_dist.log_prob(imagine_action)

    if self.actor_loss_type == 'dynamic':
      actor_loss = target

    elif self.actor_loss_type == 'reinforce':
      baseline = self.value(imagine_feat[:, :slice_idx]).mean
      advantage = (target - baseline).detach()
      actor_loss = actor_logprob[:, :slice_idx].unsqueeze(2) * advantage

    elif self.actor_loss_type == 'both':
      raise NotImplementedError

    actor_entropy = actor_dist.entropy()
    ent_scale = self.ent_scale
    actor_loss = ent_scale * actor_entropy[:, :slice_idx].unsqueeze(2) + actor_loss
    actor_loss = -(weights[:, :slice_idx] * actor_loss).mean()

    self.value.train()
    self.value.requires_grad_(True)
    imagine_value_dist = self.value(imagine_feat[:,:slice_idx].detach())
    log_prob = -imagine_value_dist.log_prob(target.detach())
    value_loss = weights[:, :slice_idx] * log_prob.unsqueeze(2)
    value_loss = value_loss.mean()
    imagine_value = imagine_value_dist.mean
  

    if global_step % self.log_every_step == 0:
      imagine_dist = Independent(OneHotCategorical(logits=imagine_state['logits']), 1)
      if self.action_dist == 'onehot':
        action_samples = imagine_action.argmax(dim=-1).float().detach()
      else:
        action_samples = imagine_action.detach()
      logs = {
        'value_loss': value_loss.detach().item(),
        'actor_loss': actor_loss.detach().item(),
        'ACT_imag_state': {k: v.detach() for k, v in imagine_state.items()},
        'ACT_imag_entropy': imagine_dist.entropy().mean().detach().item(),
        'ACT_actor_entropy': actor_entropy.mean().item(),
        'ACT_action_prob': actor_dist.mean.detach(),
        'ACT_actor_logprob': actor_logprob.mean().item(),
        'ACT_action_samples': action_samples,
        'ACT_image_discount': imagine_disc.detach(),
        'ACT_imag_value': imagine_value.squeeze(-1).detach(),
        'ACT_actor_target': target.mean().detach(),
        'ACT_target': target.squeeze(-1).detach(),
        'ACT_actor_baseline': baseline.mean().detach(),
        'ACT_imag_reward': imagine_reward.detach(),
        'ACT_imagine_idx': imagine_idx.float(),
      }
    else:
      logs = {}

    return actor_loss, value_loss, logs

  def compute_target(self, imag_feat, reward, discount_arr):
    self.slow_value.eval()
    self.slow_value.requires_grad_(False)

    value = self.slow_value(imag_feat).mean  # B*T, H, 1

    # v_t = R_{t+1} + v_{t+1}
    target = self.lambda_return(reward[:, 1:], value[:, :-1], discount_arr[:, 1:],
                                value[:, -1], self.lambda_)

    discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]), discount_arr[:, :-1]], dim=1)
    weights = torch.cumprod(discount_arr, 1).detach()  # B, T 1
    return target, weights


  def policy(self, prev_obs, obs, action, gradient_step, temp, state=None, training=True, context_len=49):
    """

    :param obs: B, C, H, W
    :param action: B, T, C
    :param gradient_step:
    :param state: B, T, C
    :param training:
    :param prior:
    :return:
    """
    obs = obs.unsqueeze(1) / 255. - 0.5 # B, T, C, H, W
    obs_emb = self.world_model.dynamic.img_enc(obs) # B, T, C
    post = self.world_model.dynamic.infer_post_stoch(obs_emb, temp, action=None) # B, T, N, C

    if state is None:
      state = post
      prev_obs = prev_obs.unsqueeze(1) / 255. - 0.5  # B, T, C, H, W
      prev_obs_emb = self.world_model.dynamic.img_enc(prev_obs)  # B, T, C
      prev_post = self.world_model.dynamic.infer_post_stoch(prev_obs_emb, temp, action=None)  # B, T, N, C

      for k, v in post.items():
        state[k] = torch.cat([prev_post[k], v], dim=1)
      s_t = state['stoch']

    else:
      s_t = torch.cat([state['stoch'], post['stoch'][:, -1:]], dim=1)[:, -context_len:]
      for k, v in post.items():
        state[k] = torch.cat([state[k], v], dim=1)[:, -context_len:]

    pred_prior = self.world_model.dynamic.infer_prior_stoch(s_t[:, :-1], temp, action)

    post_state_trimed = {}
    for k, v in state.items():
      if k in ['stoch', 'logits', 'pos', 'mean', 'std']:
        post_state_trimed[k] = v[:, 1:]
      else:
        post_state_trimed[k] = v
    post_state_trimed['deter'] = pred_prior['deter']
    post_state_trimed['o_t'] = pred_prior['o_t']

    rnn_feature = self.world_model.dynamic.get_feature(post_state_trimed, layer=self.reward_layer)
    pred_action_pdf = self.actor(rnn_feature[:, -1:].detach())

    if training:
      pred_action = pred_action_pdf.sample() # B, 1, C
    else:
      if self.action_dist == 'onehot':
        pred_action = pred_action_pdf.mean
        index = pred_action.argmax(dim=-1)[0]
        pred_action = torch.zeros_like(pred_action)
        pred_action[..., index] = 1
      else:
        pred_action = pred_action_pdf.mode

    action = torch.cat([action, pred_action], dim=1)[:, -(context_len-1):] # B, T, C

    return action, state

  def lambda_return(self, imagine_reward, imagine_value, discount, bootstrap, lambda_):
    """
    https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/algos/dreamer_algo.py
    """
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([imagine_value[:, 1:], bootstrap[:, None]], 1)
    target = imagine_reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(imagine_reward.shape[1] - 1, -1, -1))

    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:

      inp = target[:, t]
      discount_factor = discount[:, t]

      accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
      outputs.append(accumulated_reward)

    returns = torch.flip(torch.stack(outputs, dim=1), [1])
    return returns

  def update_slow_target(self, global_step):
    with torch.no_grad():
      if self.slow_update % self.slow_update_step == 0:
        self.slow_value.load_state_dict(self.value.state_dict())

      self.slow_update += 1

