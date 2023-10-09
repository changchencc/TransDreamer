import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
import pdb


class SubConv2DBlock(nn.Module):
  def __init__(self, c_in, c_out, k, s, p, num_groups,
               bias=True, non_linearity=True, weight_init='xavier',
               act='relu'):
    super().__init__()

    self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=1, padding=p, bias=bias)
    self.pix_shuffle = nn.PixelShuffle(s)

    if weight_init == 'xavier':
      nn.init.xavier_uniform_(self.conv.weight)
    else:
      nn.init.kaiming_uniform_(self.conv.weight)

    if bias:
      nn.init.zeros_(self.conv.bias)

    if num_groups > 0:
      self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out//s//s)

    if non_linearity:
      if act == 'relu':
        self.non_linear = nn.ReLU()
      else:
        self.non_linear = nn.CELU()

    self.non_linearity = non_linearity
    self.num_groups = num_groups

  def forward(self, inputs):

    o = self.conv(inputs)
    o = self.pix_shuffle(o)

    if self.num_groups > 0:
      o = self.group_norm(o)

    if self.non_linearity:
      o = self.non_linear(o)

    return o

class Conv2DBlock(nn.Module):
  def __init__(self, c_in, c_out, k, s, p, num_groups=0,
               bias=True, non_linearity=True, weight_init='xavier',
               act='relu'):
    super().__init__()

    self.net = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)

    if weight_init == 'xavier':
      nn.init.xavier_uniform_(self.net.weight)
    else:
      nn.init.kaiming_uniform_(self.net.weight)

    if bias:
      nn.init.zeros_(self.net.bias)

    if num_groups > 0:
      self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)

    if non_linearity:
      if act == 'relu':
        self.non_linear = nn.ReLU()
      elif act == 'elu':
        self.non_linear = nn.ELU()
      else:
        self.non_linear = nn.CELU()

    self.non_linearity = non_linearity
    self.num_groups = num_groups

  def forward(self, inputs):

    o = self.net(inputs)

    if self.num_groups > 0:
      o = self.group_norm(o)

    if self.non_linearity:
      o = self.non_linear(o)

    return o

class ConvTranspose2DBlock(nn.Module):
  def __init__(self, c_in, c_out, k, s, p, num_groups=0,
               bias=True, non_linearity=True, weight_init='xavier',
               act='relu'):
    super().__init__()

    self.net = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)

    if weight_init == 'xavier':
      nn.init.xavier_uniform_(self.net.weight)
    else:
      nn.init.kaiming_uniform_(self.net.weight)

    if bias:
      nn.init.zeros_(self.net.bias)

    if num_groups > 0:
      self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)

    if non_linearity:
      if act == 'relu':
        self.non_linear = nn.ReLU()
      elif act == 'elu':
        self.non_linear = nn.ELU()
      else:
        self.non_linear = nn.CELU()

    self.non_linearity = non_linearity
    self.num_groups = num_groups

  def forward(self, inputs):

    o = self.net(inputs)

    if self.num_groups > 0:
      o = self.group_norm(o)

    if self.non_linearity:
      o = self.non_linear(o)

    return o

class ResConv2DBlock(nn.Module):
  def __init__(self, c_in, c_out, num_groups,
               weight_init='xavier', act='relu'):
    super().__init__()

    self.residuel = nn.Sequential(
      nn.ReLU(),
      Conv2DBlock(c_in, c_out//2, 3, 1, 1, num_groups,
                  weight_init=weight_init, act=act),
      Conv2DBlock(c_out//2, c_out, 1, 1, 0, 0,
                  non_linearity=False, weight_init=weight_init)
    )

    self.skip = Conv2DBlock(c_in, c_out, 3, 1, 1, num_groups,
                            non_linearity=False, weight_init=weight_init)

    if num_groups > 0:
      self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
    if act == 'relu':
      self.non_linear = nn.ReLU()
    else:
      self.non_linear = nn.CELU()

  def forward(self, inputs):
    return self.non_linear(self.group_norm(self.residuel(inputs) + self.skip(inputs)))

class Linear(nn.Module):
  def __init__(self, dim_in, dim_out, bias=True, weight_init='xavier'):
    super().__init__()

    self.net = nn.Linear(dim_in, dim_out, bias=bias)

    if weight_init == 'xavier':
      nn.init.xavier_uniform_(self.net.weight)
    else:
      nn.init.kaiming_uniform_(self.net.weight)

    if bias:
      nn.init.zeros_(self.net.bias)

  def forward(self, inputs):
    return self.net(inputs)

class MLP(nn.Module):
  def __init__(self, dims, act, weight_init, output_act=None, norm=False):
    super().__init__()

    dims_in = dims[:-2]
    dims_out = dims[1:-1]

    layers = []
    for d_in, d_out in zip(dims_in, dims_out):
      layers.append(Linear(d_in, d_out, weight_init=weight_init, bias=True))
      if norm:
        layers.append(nn.LayerNorm(d_out))
      if act == 'relu':
        layers.append(nn.ReLU())
      elif act == 'elu':
        layers.append(nn.ELU())
      else:
        layers.append(nn.CELU())

    layers.append(Linear(d_out, dims[-1], weight_init=weight_init, bias=True))
    if output_act:
      if norm:
        layers.append(nn.LayerNorm(dims[-1]))
      if act == 'relu':
        layers.append(nn.ReLU())
      else:
        layers.append(nn.CELU())

    self.enc = nn.Sequential(*layers)

  def forward(self, x):
    return self.enc(x)

class GRUCell(nn.Module):
  def __init__(self, input_size, hidden_size, bias=True):
    super().__init__()
    self.gru_cell = nn.GRUCell(input_size, hidden_size)

    nn.init.xavier_uniform_(self.gru_cell.weight_ih)
    nn.init.orthogonal_(self.gru_cell.weight_hh)

    if bias:
      nn.init.zeros_(self.gru_cell.bias_ih)
      nn.init.zeros_(self.gru_cell.bias_hh)

    self.h0 = nn.Parameter(torch.randn(1, hidden_size))
    torch.nn.init.kaiming_uniform_(self.h0)

  def init_state(self, batch_size):

    return self.h0.expand(batch_size, -1)

  def forward(self, x, h):
    """
    GRU for slot attention.
    inputs:
      x: (bs, c)
      h: (bs, h)
    """

    output_shape = h.shape
    x = x.reshape(-1, x.shape[-1])
    h = h.reshape(-1, h.shape[-1])

    h = self.gru_cell(x, h)
    h = h.reshape(output_shape)

    return h

class RNNCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.rnn_cell = nn.RNNCell(input_size, hidden_size)
    nn.init.zeros_(self.rnn_cell.bias_ih)
    nn.init.zeros_(self.rnn_cell.bias_hh)

  def forward(self, x, h):
    """
    GRU for slot attention.
    inputs:
      x: (bs, N, c)
      h: (bs, N, h)
    """
    assert x.dim() == 3, 'dim of input for GRUCell should be 3'
    assert h.dim() == 3, 'dim of input for GRUCell should be 3'

    output_shape = h.shape
    x = x.reshape(-1, x.shape[-1])
    h = h.reshape(-1, h.shape[-1])

    h = self.rnn_cell(x, h)
    h = h.reshape(output_shape)

    return h

class ConvLayerNormGRUCell(nn.Module):
  def __init__(self, input_size, hidden_size, init_state=False):
    super().__init__()

    self.conv_i2h = Conv2DBlock(input_size, 2*hidden_size, 3, 1, 1, 0,
                              bias=False, non_linearity=False) # we have layernorm, bias is redundant
    self.conv_h2h = Conv2DBlock(hidden_size, 2*hidden_size, 3, 1, 1, 0,
                                bias=False, non_linearity=False)
    self.conv_i2c = Conv2DBlock(input_size, hidden_size, 3, 1, 1, 0,
                                bias=False, non_linearity=False)
    self.conv_h2c = Conv2DBlock(hidden_size, hidden_size, 3, 1, 1, 0,
                                bias=False, non_linearity=False)

    self.norm_hh = nn.GroupNorm(1, 2*hidden_size)
    self.norm_ih = nn.GroupNorm(1, 2*hidden_size)
    self.norm_c = nn.GroupNorm(1, hidden_size)
    self.norm_u = nn.GroupNorm(1, hidden_size)

    nn.init.xavier_uniform_(self.conv_i2h.net.weight)
    nn.init.xavier_uniform_(self.conv_i2c.net.weight)
    nn.init.orthogonal_(self.conv_h2h.net.weight)
    nn.init.orthogonal_(self.conv_h2c.net.weight)

    if init_state:
      self.h0 = nn.Parameter(torch.randn(1, hidden_size, 4, 4))
      torch.nn.init.kaiming_uniform_(self.h0)

  def init_state(self, shape):

    bs = shape[0]

    return self.h0.expand(bs, -1, -1, -1)

  def forward(self, x, h):
    """
    GRU for slot attention.
    inputs:
      x: (bs, C, H, W)
      h: (bs, C, H, W)
    """

    if h is None:
      h = self.init_state(x.shape)

    assert x.dim() == 4, 'dim of input for GRUCell should be 3'
    assert h.dim() == 4, 'dim of input for GRUCell should be 3'

    output_shape = h.shape

    i2h = self.conv_i2h(x)
    h2h = self.conv_h2h(h)

    logits_zr = self.norm_ih(i2h) + self.norm_hh(h2h)
    z, r = logits_zr.chunk(2, dim=1)

    i2c = self.conv_i2c(x)
    h2c = self.conv_h2c(h)

    c = (self.norm_c(i2c) + r.sigmoid() * self.norm_u(h2c)).tanh()

    h = (1. - z.sigmoid()) * h + z.sigmoid() * c

    h = h.reshape(output_shape)

    return h

class LayerNormGRUCellV2(nn.Module):
  """
  This is used in dreamerV2.
  """
  def __init__(self, input_size, hidden_size):
    super().__init__()
    input_size = input_size + hidden_size

    self.fc = Linear(input_size, 3*hidden_size, bias=False) # we have layernorm, bias is redundant

    self.layer_norm = nn.LayerNorm(3*hidden_size)

    nn.init.xavier_uniform_(self.fc.net.weight)

    self.h0 = nn.Parameter(torch.randn(1, hidden_size))
    torch.nn.init.kaiming_uniform_(self.h0)

  def init_state(self, batch_size):

    return self.h0.expand(batch_size, -1)

  def forward(self, x, h):
    """
    GRU for slot attention.
    inputs:
      x: (bs, N, c), or (bs, C)
      h: (bs, N, h), or (bs, C)
    """

    if h is None:
      h = self.init_state(x.shape)

    assert x.dim() == 3 or x.dim() == 2, 'dim of input for GRUCell should be 3 or 2'
    assert h.dim() == 3 or x.dim() == 2, 'dim of input for GRUCell should be 3 or 2'

    output_shape = h.shape
    if x.dim() == 3:
      x = x.reshape(-1, x.shape[-1])
      h = h.reshape(-1, h.shape[-1])


    logits = self.fc(torch.cat([x, h], dim=-1))
    logits = self.layer_norm(logits)

    r, c, u = logits.chunk(3, dim=-1)

    r = r.sigmoid()
    c = (r * c).tanh()
    u = (u - 1.).sigmoid()

    h = u * c + (1. - u) * h
    h = h.reshape(output_shape)

    return h

class LayerNorm(nn.Module):
  def __init__(self, normalized_shape, eps=1e-05):
    super().__init__()
    if isinstance(normalized_shape, int):
      normalized_shape = [normalized_shape]
    self.gamma = nn.Parameter(torch.Tensor(*normalized_shape))
    self.beta = nn.Parameter(torch.Tensor(*normalized_shape))
    nn.init.zeros_(self.beta)
    nn.init.ones_(self.gamma)
    self.eps = eps

  def forward(self, inpts):

    try:
      mean = inpts.mean(dim=-1, keepdim=True)
      std = inpts.std(dim=-1, keepdim=True)
      normed = (inpts - mean) / (std + self.eps).sqrt()
      o = normed * self.gamma + self.beta
    except:
      pdb.set_trace()

    return o


class LayerNormGRUCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.fc_i2h = Linear(input_size, 2*hidden_size, bias=False) # we have layernorm, bias is redundant
    self.fc_h2h = Linear(hidden_size, 2*hidden_size, bias=False)
    self.fc_i2c = Linear(input_size, hidden_size, bias=False)
    self.fc_h2c = Linear(hidden_size, hidden_size, bias=False)

    self.layer_norm_hh = nn.LayerNorm(2*hidden_size)
    self.layer_norm_ih = nn.LayerNorm(2*hidden_size)
    self.layer_norm_c = nn.LayerNorm(hidden_size)
    self.layer_norm_u = nn.LayerNorm(hidden_size)

    nn.init.xavier_uniform_(self.fc_i2h.net.weight)
    nn.init.xavier_uniform_(self.fc_i2c.net.weight)
    nn.init.orthogonal_(self.fc_h2h.net.weight)
    nn.init.orthogonal_(self.fc_h2c.net.weight)

    self.h0 = nn.Parameter(torch.randn(1, hidden_size))
    torch.nn.init.kaiming_uniform_(self.h0)

  def init_state(self, batch_size):

    return self.h0.expand(batch_size, -1)

  def forward(self, x, h):
    """
    GRU for slot attention.
    inputs:
      x: (bs, N, c), or (bs, C)
      h: (bs, N, h), or (bs, C)
    """

    if h is None:
      h = self.init_state(x.shape)

    assert x.dim() == 3 or x.dim() == 2, 'dim of input for GRUCell should be 3 or 2'
    assert h.dim() == 3 or x.dim() == 2, 'dim of input for GRUCell should be 3 or 2'

    output_shape = h.shape
    if x.dim() == 3:
      x = x.reshape(-1, x.shape[-1])
      h = h.reshape(-1, h.shape[-1])


    i2h = self.fc_i2h(x)
    h2h = self.fc_h2h(h)

    logits_zr = self.layer_norm_ih(i2h) + self.layer_norm_hh(h2h)
    z, r = logits_zr.chunk(2, dim=-1)

    i2c = self.fc_i2c(x)
    h2c = self.fc_h2c(h)

    c = (self.layer_norm_c(i2c) + r.sigmoid() * self.layer_norm_u(h2c)).tanh()

    h_new = (1. - z.sigmoid()) * h + z.sigmoid() * c

    h_new = h_new.reshape(output_shape)

    return h_new

class ConvLSTMCell(nn.Module):
  def __init__(self, cfg, input_size, hidden_size, spatial_size, k=3, p=1):
    super().__init__()

    weight_init = cfg.arch.weight_init

    self.conv_ = Conv2DBlock(input_size + hidden_size, 4*hidden_size, k, 1, p, 0,
                                bias=False, non_linearity=False, weight_init=weight_init)

    self.bias = nn.Parameter(torch.zeros(1, 4*hidden_size, 1, 1), requires_grad=True)

    self.h0, self.c0 = self.init_state(hidden_size, spatial_size, weight_init)

  def init_state(self, hidden_size, spatial_size, weight_init):

    h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)
    c0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

    nn.init.zeros_(h0)
    nn.init.zeros_(c0)

    return nn.Parameter(h0, requires_grad=True), nn.Parameter(c0, requires_grad=True)

  def forward(self, x, state):
    """
    LSTM for global attention decoder.
    """

    if state is None:
      bs = x.shape[0]

      h, c = self.h0, self.c0
      h = h.expand(bs, -1, -1, -1)
      c = c.expand(bs, -1, -1, -1)

    else:
      h, c = state

    logits = self.conv_(torch.cat([x, h], dim=1)) + self.bias

    f, i, o, g = logits.chunk(4, dim=1)

    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
    h = o.sigmoid() * c.tanh()

    return h, c

class ConvLayerNormLSTMCell(nn.Module):
  def __init__(self, cfg, input_size, hidden_size, spatial_size, k=3, p=1):
    super().__init__()

    weight_init = cfg.arch.weight_init

    self.conv_i2h = Conv2DBlock(input_size, 4*hidden_size, k, 1, p, 0,
                              bias=False, non_linearity=False, weight_init=weight_init)
    self.conv_h2h = Conv2DBlock(hidden_size, 4*hidden_size, k, 1, p, 0,
                            bias=False, non_linearity=False, weight_init=weight_init)

    self.norm_h = nn.GroupNorm(1, 4*hidden_size)
    self.norm_i = nn.GroupNorm(1, 4*hidden_size)
    self.norm_c = nn.GroupNorm(1, hidden_size)

    self.bias = nn.Parameter(torch.zeros(1, 4*hidden_size, 1, 1), requires_grad=True)

    self.h0, self.c0 = self.init_state(hidden_size, spatial_size, weight_init)

  def init_state(self, hidden_size, spatial_size, weight_init):

    h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)
    c0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

    nn.init.zeros_(h0)
    nn.init.zeros_(c0)

    return nn.Parameter(h0, requires_grad=True), nn.Parameter(c0, requires_grad=True)

  def forward(self, x, state):
    """
    LSTM for global attention decoder.
    """

    if state is None:
      bs = x.shape[0]

      h, c = self.h0, self.c0
      h = h.expand(bs, -1, -1, -1)
      c = c.expand(bs, -1, -1, -1)

    else:
      h, c = state

    i2h = self.conv_i2h(x)
    h2h = self.conv_h2h(h)

    logits = self.norm_i(i2h) + self.norm_h(h2h) + self.bias
    # logits = i2h + h2h + self.bias

    f, i, o, g = logits.chunk(4, dim=1)

    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
    h = o.sigmoid() * self.norm_c(c).tanh()

    return h, c

class LayerNormLSTMCell(nn.Module):
  def __init__(self, cfg, input_size, hidden_size):
    super().__init__()

    self.fc_i2h = Linear(input_size, 4*hidden_size, weight_init=cfg.arch.weight_init, bias=False)
    self.fc_h2h = Linear(hidden_size, 4*hidden_size, weight_init=cfg.arch.weight_init, bias=False)

    self.layer_norm_h = nn.LayerNorm(4*hidden_size)
    self.layer_norm_i = nn.LayerNorm(4*hidden_size)
    self.layer_norm_c = nn.LayerNorm(hidden_size)

    self.bias = nn.Parameter(torch.zeros(4*hidden_size), requires_grad=True)

    self.h0, self.c0 = self.init_state(hidden_size)

  def init_state(self, hidden_size):

    h0 = torch.randn(1, hidden_size)
    c0 = torch.randn(1, hidden_size)

    nn.init.zeros_(h0)
    nn.init.zeros_(c0)

    return nn.Parameter(h0, requires_grad=True), nn.Parameter(c0, requires_grad=True)

  def forward(self, x, state):
    """
    LSTM for global attention decoder.
    """

    h, c = state

    i2h = self.fc_i2h(x)
    h2h = self.fc_h2h(h)

    logits = self.layer_norm_i(i2h) + self.layer_norm_h(h2h) + self.bias

    f, i, o, g = logits.chunk(4, dim=-1)

    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
    h = o.sigmoid() * self.layer_norm_c(c).tanh()

    return h, c


class GroupLSTMCell(nn.Module):
  def __init__(self, cfg, input_size, hidden_size, num_units):
    super().__init__()

    self.hidden_size = hidden_size
    self.input_size = input_size
    self.num_units = num_units

    self.i2h = Linear(num_units * input_size, 4 * num_units * hidden_size, weight_init=cfg.arch.weight_init, bias=False)
    self.h2h = Linear(num_units * hidden_size, 4 * num_units * hidden_size, weight_init=cfg.arch.weight_init, bias=False)

    self.bias = nn.Parameter(torch.zeros(1, num_units, 4 * hidden_size), requires_grad=True)
    self.h0, self.c0 = self.init_states()

  def init_states(self):
    h = torch.randn(1, self.num_units, self.hidden_size)
    c = torch.randn(1, self.num_units, self.hidden_size)
    nn.init.zeros_(h)
    nn.init.zeros_(c)

    return nn.Parameter(h, requires_grad=True), nn.Parameter(c, requires_grad=True)

  def forward(self, x, h, c):
    """
    x: bs, num_units, C
    h, c: bs, num_units, H
    """

    i2h = self.i2h(x.reshape(-1, self.num_units * self.input_size))
    h2h = self.h2h(h.reshape(-1, self.num_units * self.hidden_size))

    i2h = i2h.reshape(-1, self.num_units, 4 * self.hidden_size)
    h2h = h2h.reshape(-1, self.num_units, 4 * self.hidden_size)

    logits = i2h + h2h + self.bias

    f, i, o, g = logits.chunk(4, dim=-1)

    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
    h = o.sigmoid() * c.tanh()

    return h, c

class MyRelaxedBernoulli(RelaxedBernoulli):
  def __init__(self, temp, logits=None, probs=None):
    super(MyRelaxedBernoulli, self).__init__(temp, probs=probs, logits=logits)
    """
    re-write the rsample() api.
    """
    if logits is None:
      self.device = probs.device
    if probs is None:
      self.device = logits.device

  def rsample(self, shape=None, eps=1e-15):
    if shape is None:
      shape = self.probs.shape
    uniforms = torch.rand(shape, dtype=self.logits.dtype, device=self.device)
    uniforms = torch.clamp(uniforms, eps, 1. - eps)
    samples = ((uniforms).log() - (-uniforms).log1p() + self.logits) / self.temperature
    if torch.isnan(samples).any():
      pdb.set_trace()
    return samples

  def log_prob(self, values):
    diff = self.logits - values.mul(self.temperature)

    l_p = self.temperature.log() + diff - 2 * diff.exp().log1p()
    if torch.isinf(l_p).any():
      pdb.set_trace()
    return l_p

def linear_annealing(step, start_step, end_step, start_value, end_value):
  """
  Linear annealing

  :param x: original value. Only for getting device
  :param step: current global step
  :param start_step: when to start changing value
  :param end_step: when to stop changing value
  :param start_value: initial value
  :param end_value: final value
  :return:
  """
  if step <= start_step:
    x = start_value
  elif start_step < step < end_step:
    slope = (end_value - start_value) / (end_step - start_step)
    x = start_value + slope * (step - start_step)
  else:
    x = end_value

  return x

def up_and_down_linear_schedule(step, start_step, mid_step, end_step,
                                start_value, mid_value, end_value):
  if start_step < step <= mid_step:
    slope = (mid_value - start_value) / (mid_step - start_step)
    x = start_value + slope * (step - start_step)
  elif mid_step < step < end_step:
    slope = (end_value - mid_value) / (end_step - mid_step)
    x = mid_value + slope * (step - mid_step)
  elif step >= end_step:
    x = end_value
  else:
    x = start_value

  return x

class GatedCNN(nn.Module):

  def __init__(self, cfg, input_size, hidden_size, spatial_size):
    super().__init__()


    if spatial_size == 1:
      k = 1
      p = 0
    else:
      k = 3
      p = 1

    weight_init = cfg.arch.weight_init
    self.conv_h = Conv2DBlock(input_size+hidden_size, hidden_size * 2, k, 1, p, 0,
                              non_linearity=False, weight_init=weight_init)

  def init_state(self, hidden_size, spatial_size):

    h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

    nn.init.zeros_(h0)

    return nn.Parameter(h0, requires_grad=True)

  def forward(self, x, h):

    h = self.conv_h(torch.cat([x, h], dim=1))

    h1, h2 = torch.chunk(h, 2, dim=1)
    h = torch.tanh(h1) * torch.sigmoid(h2)

    return h

if __name__ == '__main__':
  dims = [64, 64, 4]
  net = MLP(dims, nn.ReLU)
  pdb.set_trace()







