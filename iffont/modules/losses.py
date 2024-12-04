import torch
import torch.nn as nn
from torch.nn import functional as F


def z_loss(logits:torch.Tensor) -> torch.Tensor:
  '''z-loss in Small-scale proxies for large-scale Transformer training instabilities
  logits: [bs, block_size, vocab_size]
  '''
  z = logits.exp().sum(dim=-1)
  loss = z.log_().pow_(2).mean()
  return loss


def sa(a:torch.Tensor, p:torch.Tensor, n:torch.Tensor):
  '''special aware loss'''
  return F.triplet_margin_loss(a, p, n)


def sq(logits:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
  logits = logits.reshape(-1, logits.size(-1))
  target = target.flatten()
  return F.cross_entropy(logits, target)


def ic(pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
  '''idc count average loss  
  args:  
    pred: (bs, n_idc), e.g. [[0, 2, 1, 6, 0, 0, ..., 1], [...]]  
    target: (bs, n_idc)  
  '''

  # loss = F.mse_loss(pred, target)
  loss = F.l1_loss(pred, target)
  # loss = F.huber_loss(pred, target)
  return loss


def bidirectional_kl(p, q, pad_mask=None):
  p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
  q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
  
  # pad_mask is for seq-level tasks
  if pad_mask is not None:
    p_loss.masked_fill_(pad_mask, 0.)
    q_loss.masked_fill_(pad_mask, 0.)

  # You can choose whether to use function "sum" and "mean" depending on your task
  p_loss = p_loss.sum()
  q_loss = q_loss.sum()

  loss = (p_loss + q_loss) / 2
  return loss


class ArcFace(nn.Module):
  """ https://arxiv.org/pdf/1801.07698v1.pdf
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/partial_fc_v2.py#L10 +
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py#L60 /
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py#L10
  """

  def __init__(self, num_classes, s=64.0, margin=0.5):
    super().__init__()
    self.num_classes = num_classes
    # AdaCos: s = sqrt(2)*(log(C-1))
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/186
    self.s = s  # the inverse of "softmax temperature", higher scale parameter will result in larger gradients during training
    self.margin = margin  # increasing the margin results in a bigger separation of classes
    self.weight = nn.Parameter(torch.normal(0, 0.01, (num_classes, num_classes)))

  def forward(self, logits: torch.Tensor, labels: torch.Tensor):
    '''
      logits: shape of (bs, len, c)
      labels: shape of (bs, len)
    '''

    logits = F.linear(F.normalize(logits), F.normalize(self.weight))
    logits.clamp_(-1, 1)
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.flatten()

    # index = torch.where(labels != -1)[0]
    index = torch.arange(labels.shape[0])
    index = (index, labels, )

    with torch.no_grad():
      logits.arccos_()
      logits[index] += self.margin
      logits.cos_()
    logits = logits * self.s

    loss = F.cross_entropy(logits, labels)
    return loss


def sup_cl(features, labels=None, mask=None, temperature=0.07, contrast_mode='all', base_temperature=0.07):
  """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. 
    Compute loss for model. If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
    It also supports the unsupervised contrastive loss in SimCLR

    https://arxiv.org/pdf/2002.05709.pdf
    Args:
      features: hidden vector of shape [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        has the same class as sample i. Can be asymmetric.
    Returns:
      A loss scalar.
  """

  features = F.normalize(features, p=2, dim=2)
  device = features.device
  batch_size, contrast_count = features.shape[0], features.shape[1]

  if len(features.shape) < 3:
    raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
  if len(features.shape) > 3:
    features = features.view(batch_size, contrast_count, -1)

  if labels is not None and mask is not None:
    raise ValueError('Cannot define both `labels` and `mask`')
  elif labels is None and mask is None:
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
  elif labels is not None:
    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
      raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().to(device)
  else:
    # mask.shape: (bs, bs)
    mask = mask.float().to(device)

  contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
  if contrast_mode == 'one':
    anchor_feature = features[:, 0]
    anchor_count = 1
  elif contrast_mode == 'all':
    anchor_feature = contrast_feature
    anchor_count = contrast_count
  else:
    raise ValueError('Unknown mode: {}'.format(contrast_mode))

  # compute logits (bs*anchor_count, bs*contrast_count)
  anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)
  # for numerical stability
  logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
  logits = anchor_dot_contrast - logits_max.detach()

  mask = mask.repeat(anchor_count, contrast_count)
  # mask-out self-contrast cases
  logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
  mask = mask * logits_mask

  # compute log_prob
  exp_logits = torch.exp(logits) * logits_mask
  log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

  # compute mean of log-likelihood over positive
  mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
  loss = -(temperature / base_temperature) * mean_log_prob_pos
  loss = loss.view(anchor_count, batch_size).mean()

  return loss


class PairConLoss(nn.Module):
  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, f1, f2, mask=None, label=None):

    assert mask is not None or label is not None
    f1 = F.normalize(f1.reshape(f1.shape[0], -1), dim=1)
    f2 = F.normalize(f2.reshape(f2.shape[0], -1), dim=1)

    logits = torch.einsum('nc,mc->nm', f1, f2) / self.temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    if label is None:
      label = torch.nonzero(mask)[:, 1]
      assert len(label) == f1.shape[0]

    loss = sq(logits, label) * (2 * self.temperature)
    return loss
