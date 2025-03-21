from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropKeyMask(nn.Module):

  def __init__(self, mask_ratio=0.1, is_causal=False, dtype=bool) -> None:
    super().__init__()
    self.p = 1 - mask_ratio
    self.is_causal = is_causal
    self.dtype = dtype
    self.delta = -1e12

  def _check_row(self, mask: torch.Tensor):
    nr, nc = mask.shape[-2:]
    if self.is_causal:
      mask[..., 0, 0] = 1

    droped_row = mask.sum(-1, dtype=torch.bool).flatten()
    droped_row = droped_row.logical_not_().argwhere().flatten()
    if droped_row.shape[-1] > 0:
      droped_col = droped_row % nc
      mask.reshape(-1, nc)[droped_row, droped_col] = True

  def __call__(self, *size, device=None) -> torch.Tensor:
    if not self.training and self.p == 1:
      return None
    mask = torch.ones(size, dtype=torch.bool, device=device)
    if self.training:
      mask.bernoulli_(self.p)
    if self.is_causal:
      mask.tril_(diagonal=0)
    self._check_row(mask)

    if self.dtype != bool:
      mask = (~mask).type(self.dtype).mul_(self.delta)
    return mask


def spectral_norm(module):
  """ init & apply spectral norm """
  nn.init.xavier_uniform_(module.weight, 2**0.5)
  if hasattr(module, 'bias') and module.bias is not None:
    module.bias.data.zero_()

  return nn.utils.spectral_norm(module)


def dispatcher(dispatch_fn):

  def decorated(key, *args):
    if callable(key):
      return key
    if key is None:
      key = 'none'
    return dispatch_fn(key, *args)

  return decorated


@dispatcher
def norm_dispatch(norm):
  return {
      'none': nn.Identity,
      'in': partial(nn.InstanceNorm2d, affine=False),  # false as default
      'bn': nn.BatchNorm2d,
  }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
  return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
  return {
      "none": nn.Identity,
      "relu": nn.ReLU,
      "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
  }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
  return {"zero": nn.ZeroPad2d, "replicate": nn.ReplicationPad2d, "reflect": nn.ReflectionPad2d}[pad_type.lower()]


class ConvBlock(nn.Module):
  """ pre-active conv block """

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none', activ='relu', bias=True, upsample=False, downsample=False, w_norm='none', pad_type='zero', dropout=0., size=None):
    # 1x1 conv assertion
    if kernel_size == 1:
      assert padding == 0
    super().__init__()
    self.C_in = C_in
    self.C_out = C_out

    activ = activ_dispatch(activ, norm)
    norm = norm_dispatch(norm)
    w_norm = w_norm_dispatch(w_norm)
    pad = pad_dispatch(pad_type)
    self.upsample = upsample
    self.downsample = downsample

    self.norm = norm(C_in)
    self.activ = activ()

    if dropout > 0.:
      self.dropout = nn.Dropout2d(p=dropout)
    self.pad = pad(padding)
    self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

  def forward(self, x):
    x = self.norm(x)
    x = self.activ(x)
    if self.upsample:
      x = F.interpolate(x, scale_factor=2)
    if hasattr(self, 'dropout'):
      x = self.dropout(x)
    x = self.conv(self.pad(x))
    if self.downsample:
      x = F.avg_pool2d(x, 2)
    return x


class ResBlock(nn.Module):
  """ Pre-activate ResBlock with spectral normalization """

  def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False, norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0., scale_var=False):
    assert not (upsample and downsample)
    super().__init__()
    w_norm = w_norm_dispatch(w_norm)
    self.C_in = C_in
    self.C_out = C_out
    self.upsample = upsample
    self.downsample = downsample
    self.scale_var = scale_var

    self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ, upsample=upsample, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
    self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ, w_norm=w_norm, pad_type=pad_type, dropout=dropout)

    if C_in != C_out or upsample or downsample:
      self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

  def forward(self, x):
    out = x

    out = self.conv1(out)
    out = self.conv2(out)

    if self.downsample:
      out = F.avg_pool2d(out, 2)

    # skip-con
    if hasattr(self, 'skip'):
      if self.upsample:
        x = F.interpolate(x, scale_factor=2)
      x = self.skip(x)
      if self.downsample:
        x = F.avg_pool2d(x, 2)

    out = out + x
    if self.scale_var:
      out = out / np.sqrt(2)
    return out
