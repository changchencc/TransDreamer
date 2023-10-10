
# modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import Linear



class GRUGatingMechanism(torch.nn.Module):
  def __init__(self, d_input, bg=0.1):
    super().__init__()
    self.Wr = Linear(d_input, d_input, bias=False)
    self.Ur = Linear(d_input, d_input, bias=False)
    self.Wz = Linear(d_input, d_input, bias=False)
    self.Uz = Linear(d_input, d_input)
    self.Wg = Linear(d_input, d_input, bias=False)
    self.Ug = Linear(d_input, d_input, bias=False)
    self.bg = bg

    self.sigmoid = torch.nn.Sigmoid()
    self.tanh = torch.nn.Tanh()

  def forward(self, x, y):
    r = self.sigmoid(self.Wr(y) + self.Ur(x))
    z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
    h = self.tanh(self.Wg(y) + r * self.Ug(x))
    g = torch.mul(1 - z, x) + torch.mul(z, h)
    return g

class PositionalEmbedding(torch.nn.Module):
  def __init__(self, dim):
    super(PositionalEmbedding, self).__init__()

    self.dim = dim
    inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
    self.register_buffer("inv_freq", inv_freq)

  def forward(self, positions):
    sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb[:, None, :]

class PositionwiseFF(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    d_model = cfg.d_model
    d_inner = cfg.d_ff_inner
    dropout = cfg.dropout
    self.pre_lnorm = cfg.pre_lnorm

    self.CoreNet = nn.Sequential(
      Linear(d_model, d_inner),
      nn.ReLU(inplace=True),
      Linear(d_inner, d_model),
      nn.Dropout(dropout)
    )

    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, inp):

    if self.pre_lnorm:
      ##### layer normalization + positionwise feed-forward
      output = self.CoreNet(self.layer_norm(inp))

    else:
      ##### positionwise feed-forward
      core_out = self.CoreNet(inp)

      ##### layer normalization
      output = self.layer_norm(core_out)

    return output

class MultiheadAttention(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    d_model = cfg.d_model
    n_head = cfg.num_heads
    d_inner = cfg.d_inner
    dropout = cfg.dropout
    dropatt = cfg.dropatt
    pre_lnorm = cfg.pre_lnorm

    self.d_inner = d_inner
    self.n_head = n_head

    self.q_net = Linear(d_model, d_inner * n_head, bias=False)
    self.k_net = Linear(d_model, d_inner * n_head, bias=False)
    self.v_net = Linear(d_model, d_inner * n_head, bias=False)
    self.out_net = Linear(d_inner * n_head, d_model, bias=False)

    self.drop = nn.Dropout(dropout)
    self.dropatt = nn.Dropout(dropatt)
    self.layer_norm = nn.LayerNorm(d_model)

    self.scale = 1 / (d_inner ** 0.5)

    self.pre_lnorm = pre_lnorm

  def forward(self, q, k, v, attn_mask=None):
    """

    :param inpts: T, B, D
    :param pos_emb: T, B, D
    :param attn_mask: T, T
    pad_mask: B, T
    :return:
    """

    T_q, bsz = q.shape[:2]
    T_k, bsz = k.shape[:2]

    if self.pre_lnorm:
      w_head_q = self.q_net(self.layer_norm(q))
      w_head_k = self.k_net(self.layer_norm(k))
      w_head_v = self.v_net(self.layer_norm(v))
    else:
      w_head_q = self.q_net(q)
      w_head_k = self.k_net(k)
      w_head_v = self.v_net(v)

    w_head_q = w_head_q.view(T_q, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head
    w_head_k = w_head_k.view(T_k, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head
    w_head_v = w_head_v.view(T_k, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head

    attn_score = torch.einsum('ibnd,jbnd->ijbn', (w_head_q, w_head_k)) * self.scale

    #### compute attention probability
    if attn_mask is not None:
      attn_score = attn_score.float().masked_fill(
            attn_mask[:, :, None, None].bool(), -float('inf')).type_as(attn_score)

    # [qlen x klen x bsz x n_head]
    attn_prob = F.softmax(attn_score, dim=1)
    attn_prob = self.dropatt(attn_prob)

    #### compute attention vector
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

    # [qlen x bsz x n_head x d_head]
    attn_vec = attn_vec.contiguous().view(
      attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_inner)

    ##### linear projection
    attn_out = self.out_net(attn_vec)
    attn_out = self.drop(attn_out)

    if self.pre_lnorm:
      ##### residual connection
      output = attn_out
    else:
      ##### residual connection + layer normalization
      output = self.layer_norm(attn_out)

    return output

class TransformerEncoderLayer(nn.Module):

  def __init__(self, cfg):
    super().__init__()

    self.mah = MultiheadAttention(cfg)
    self.pos_ff = PositionwiseFF(cfg)

    self.gating = cfg.gating
    if self.gating:
      self.gate1 = GRUGatingMechanism(cfg.d_model)
      self.gate2 = GRUGatingMechanism(cfg.d_model)

  def forward(self, inpts, attn_mask=None):

    src2 = self.mah(inpts, inpts, inpts, attn_mask=attn_mask)
    if self.gating:
      src = self.gate1(inpts, src2)
    else:
      src = inpts + src2

    src2 = self.pos_ff(src)
    if self.gating:
      src = self.gate2(src, src2)
    else:
      src = src + src2

    return src

class Transformer(nn.Module):

  def __init__(self, cfg):
    super().__init__()

    d_model = cfg.d_model
    n_layers = cfg.n_layers
    dropout = cfg.dropout
    self.d_model = d_model
    self.n_layers = n_layers
    self.gating = cfg.gating
    self.last_ln = cfg.last_ln

    self.pos_embs = PositionalEmbedding(d_model)
    self.drop = torch.nn.Dropout(dropout)

    self.layers = torch.nn.ModuleList(
      [TransformerEncoderLayer(cfg) for _ in range(n_layers)]
    )

    if self.last_ln:
      self.ln = nn.LayerNorm(d_model)

  def _generate_square_subsequent_mask(self, T, H, W, device):
    N = H * W
    mask = (torch.triu(torch.ones(T, T,
                                  device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-1e10')).masked_fill(
      mask == 1, float(0.0))

    mask = torch.repeat_interleave(mask, N, dim=0)
    mask = torch.repeat_interleave(mask, N, dim=1)

    return mask

  def forward(self, z, actions):
    B, T, D, H, W = z.shape

    attn_mask = self._generate_square_subsequent_mask(T, H, W, z.device) # T, T

    # (T, 1, d_model)
    pos_ips = torch.arange(T*H*W, dtype=torch.float).to(z.device)
    pos_embs = self.drop(self.pos_embs(pos_ips))

    if actions is None:

      z = rearrange(z, 'b t d h w -> (t h w) b d')
      encoder_inp = z + pos_embs

    else:
      z = rearrange(z, 'b t d h w -> (t h w) b d')
      actions = rearrange(actions, 'b t d -> t b d')

      encoder_inp = self.input_embedding(z) + pos_embs
      action_emb = self.action_embedding(actions)
      action_emb = torch.repeat_interleave(action_emb, H*W, dim=0)
      encoder_inp += action_emb

    # T, B, d_model
    output = encoder_inp
    output_list = []
    for i, layer in enumerate(self.layers):
      output = layer(output, attn_mask=attn_mask) # T, B, D

      output_list.append(output)

    output = torch.stack(output_list, dim=1) # T, L, B, D

    output = rearrange(output,
                       '(t h w) l b d -> b t l d h w',
                       h=H, w=W)
    return output

