from functools import partial
import random
import copy
import math
import itertools

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torchvision.models import mobilenetv3
from torchvision.models import efficientnet
from torchvision.ops.misc import Conv2dNormActivation

from data.adapter import VQAdapter
from data import cn, valid_characters
from modules.blocks import ConvBlock, ResBlock, DropKeyMask
from util import utils


class IDSMutator:
  def __init__(self, num: int = 7, most_vary: int = 3, cache_size: None | int = None) -> None:
    self.n_variation = num
    self.most_vary = most_vary
    self._cache = dict()
    if cache_size is None:
      cache_size = self.n_variation
    self._cache_size = max(0, min(self.n_variation, cache_size))

  @staticmethod
  def add(ids: list) -> bool:
    pos = random.randint(0, len(ids))
    ids.insert(pos, random.choice(cn.IDS_VOCABULARY))
    return True

  @staticmethod
  def delete(ids: list) -> bool:
    # 删无可删
    if len(ids) == 0:
      return False
    pos = random.randint(0, len(ids) - 1)
    ids.pop(pos)
    return True

  @staticmethod
  def alter(ids: list) -> bool:
    if len(ids) == 0:
      return False
    pos = random.randint(0, len(ids) - 1)
    ids[pos] = random.choice(cn.IDS_VOCABULARY)
    return True

  def __query_cache(self, ids: str):
    if ids not in self._cache:
      self._cache[ids] = []
    return self._cache[ids]

  def __update_cache(self, ids: str, vary: list):
    if self._cache_size == 0:
      return
    self._cache[ids] = vary[-self._cache_size:]

  def vary_ids(self, ids: str) -> list[str]:
    vary = [ids]
    vary.extend(self.__query_cache(ids))
    length = len(ids)
    num2change = tuple(range(1, self.most_vary + 1))
    funcs = (
        self.add,
        self.delete,
        self.alter,
    )

    for _ in range(self.n_variation + 1 - len(vary)):
      num = min(int(length // 10), self.most_vary - 1)
      w = [1] * self.most_vary
      w[num] = 10 - length%10
      num = random.choices(num2change, weights=w, k=1)[0]

      temp = list(ids)
      w = 0
      while True:
        if random.choice(funcs)(temp):
          w += 1
        if w < num:
          continue
        (len(temp) == 0 or temp == list(ids)) and self.add(temp)
        break
      vary.append(''.join(temp))

    self.__update_cache(ids, vary)
    return vary


class _SpecialTokenGenerator:

  def __init__(self, *tokens) -> None:
    self.__data = dict()
    self.__count = 0
    for t in tokens:
      self.__data[t] = self.__count
      self.__count += 1

  def __len__(self):
    return self.__count

  def __getattr__(self, name):
    assert name in self.__data, f'{name} is not a valid special token'
    return self.__data[name]


class SequenceEncoder(nn.Module):
  def __init__(self, vocabulary: list[str], max_len: int, n_embd: int):
    super().__init__()
    self.special_tokens = _SpecialTokenGenerator('pad', 'sep',)
    self.vocabulary_maps = tuple((dict() for _ in range(len(vocabulary))))

    count = len(self.special_tokens)
    for i, v in enumerate(vocabulary):
      vocab = self.vocabulary_maps[i]
      for token in v:
        if token in vocab:
          continue
        vocab[token] = count
        count += 1

    self.num_tokens = count
    self.embedding = nn.Embedding(count, n_embd)
    self.max_len = max_len

  def set_device(self, device: torch.device):
    self.device = device

  def embed(self, sequence: str | tuple[str], vocab_idx: None | int = None) -> torch.Tensor:
    tokens = [self.special_tokens.pad] * self.max_len
    count = 0
    if isinstance(vocab_idx, int):
      vocab = self.vocabulary_maps[vocab_idx]
    if isinstance(sequence, str):
      sequence = (sequence,)

    for i, seq in enumerate(sequence):
      if vocab_idx is None:
        vocab = self.vocabulary_maps[i]
      for s in seq:
        tokens[count] = vocab[s]
        count += 1
      tokens[count] = self.special_tokens.sep
      count += 1
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)

    return self.embedding(tokens)

  def forward(self, sequence_list: tuple) -> torch.Tensor:
    res = []
    for sequences in zip(*sequence_list):
      res.append(self.embed(sequences))
    return torch.stack(res, dim=0)


class IDSEncoder(nn.Module):
  def __init__(self, max_len: int, n_embd: int, corpus='', input_mode='ids', ids_mode='all'):
    super().__init__()
    assert input_mode in {'ch', 'ids'}
    assert ids_mode in {'all', 'radical', 'stroke'}
    assert isinstance(cn.IDC, tuple), 'cn.IDC must be tuple (orderly)'

    self.input_mode = input_mode
    self.ids_mode = ids_mode
    self.special_tokens = ('pad', 'sep', ) if ids_mode == 'all' else ('pad', )
    self.vocabulary_map = dict()
    self.max_len = max_len
    corpus = corpus or valid_characters.valid_ch
    self.r_map, _, self.r_base = cn.resolve_IDS_babelstone(corpus, 'radical')
    self.s_map, _, self.s_base = cn.resolve_IDS_babelstone(corpus, 'stroke')

    count = 0
    if ids_mode == 'all':
      tmp = itertools.chain(self.r_base.keys(), self.s_base.keys())
    else:
      tmp = getattr(self, f'{ids_mode[0]}_base').keys()
    tmp = sorted(set(tmp))
    tmp = itertools.chain(self.special_tokens, cn.IDC, tmp)
    for token in tmp:
      if token in self.vocabulary_map:
        continue
      self.vocabulary_map[token] = count
      count += 1
    
    self.num_tokens = count
    self.embedding = nn.Embedding(count, n_embd)
    self.embedding2 = nn.Embedding(count, n_embd)
    self.ids_map = dict()
    for i, token in enumerate(cn.IDS_VOCABULARY[len(cn.IDC):]):
      self.ids_map[token] = i
    # self.ids_mutator = IDSMutator(n_variation, cache_size=n_variation - 1)
    # encode_layer = nn.TransformerEncoderLayer(n_embd, nhead=8, dim_feedforward=n_embd * 2, batch_first=True)
    # self.transformer = nn.ModuleDict(dict(
    #     h=nn.TransformerEncoder(encode_layer, num_layers=4),
    #     pe=nn.Embedding(max_len, n_embd),
    # ))

  def set_device(self, device: torch.device):
    self.device = device
    self.to(device)

  def query_ids(self, ch:str, ids_mode=None) -> tuple[tuple[str]]:
    ids_mode = ids_mode or self.ids_mode

    s, r = None, None
    if ids_mode != 'radical':
      s = self.s_map[ch]
      s = self.triplet_equal_ids(s)
    if ids_mode != 'stroke':
      r = self.r_map[ch]
      r = self.triplet_equal_ids(r)
    
    if s is None:
      ids = r
    elif r is None:
      ids = s
    else:
      ids = []
      for i1 in r:
        i1 += ('sep', )
        for i2 in s:
          t = i1
          t += i2
          ids.append(t)
      ids = tuple(ids)
      
    return ids
  
  def triplet_equal_ids(self, ids: tuple) -> tuple[tuple]:
    level, _ = self.level_analysis(ids, depth=2)
    idc = None
    match level:
      case '⿲', a, b, c:
        idc = 0
      case '⿰', ('⿰', *_), ('⿰', *_):
        ...
      case '⿰', ('⿰', a, b), c:
        idc = 0
      case '⿰', a, ('⿰', b, c):
        idc = 0
      case '⿳', a, b, c:
        idc = 1
      case '⿱', ('⿱', *_), ('⿱', *_):
        ...
      case '⿱', ('⿱', a, b), c:
        idc = 1
      case '⿱', a, ('⿱', b, c):
        idc = 1

    if idc is None:
      return (ids, )
    t2, t3 = (('⿰', '⿲'), ('⿱', '⿳'))[idc]

    ret = lambda x: utils.chain_sequence(*x) if isinstance(x, tuple) else x
    a, b, c = tuple(map(ret, (a, b, c)))
    ret = (
      tuple(utils.chain_sequence(t3, a, b, c)),
      tuple(utils.chain_sequence(t2, a, t2, b, c)),
      tuple(utils.chain_sequence(t2, t2, a, b, c)),
    )
    return ret

  @staticmethod
  def level_analysis(ids, depth=-1, _d=1) -> tuple:

    l_ids = len(ids)
    if l_ids <= 1:
      return (ids, l_ids, )

    main_idc = ids[0]
    assert main_idc in cn.IDC, f'illegal ids {ids}'
    level = [main_idc]
    i = 1

    while i < l_ids:
      if len(level) == cn.N_IDC_COMPS[main_idc]+1:
        break
      if ids[i] in cn.IDC:
        new_level, new_i = IDSEncoder.level_analysis(ids[i:], depth, _d+1)
        if depth > 0 and _d >= depth:
          new_level = tuple(utils.chain_sequence(*new_level))
        level.append(new_level)
        i += new_i
      else:
        level.append(ids[i])
        i += 1
  
    assert len(level) == cn.N_IDC_COMPS[main_idc]+1, f'illegal {ids=}, {level=}'
    return tuple(level), i

  def count_idc(self, ids_list: list[tuple]) -> torch.Tensor:
    res = []
    for ids in ids_list:
      for c in filter(lambda x: x not in cn.IDC, ids):
        ic[self.ids_map[c]] = 1
        ic[self.ids_map[c]] += 1
      ic = torch.as_tensor(ic, device=self.device)
      res.append(ic)
    res = torch.stack(res, dim=0)
    return res

  def coverage(self, targets:tuple, sources:tuple) -> torch.Tensor:
    def count(target, source):
      ti = match_cnt = 0
      while True:
        if ti == ti_max:
          break
        tc = target[ti]
        if tc not in cn.IDC:
          ti += 1
          continue

        si, si_max = 0, len(source)
        while True:
          if si >= si_max:
            ti += 1
            break
          if source[si] != tc:
            si += 1
            continue
      
          ti2, si2 = ti, si
          while ti2<ti_max and si2<si_max and target[ti2] == source[si2]:
            ti2 += 1
            si2 += 1

          if ti2==ti_max or target[ti2] in cn.IDC:
            match_cnt += ti2-ti
            ti = ti2
            break
          si += 1
      return match_cnt

    def get_ids(x):
      if self.input_mode == 'ch':
        ids = self.query_ids(x, 'stroke')[0]
      else:
        ids = x
      return ids
    
    res = []
    for t, source in zip(targets, sources):
      res_row = []
      t_ids = get_ids(t)
      ti_max = len(t_ids)
      for s in source:
        s_ids = get_ids(s)
        cnt = count(t_ids, s_ids)
        res_row.append(cnt / ti_max)
      res.append(res_row)
    res = torch.as_tensor(res, device=self.device)  # (bs, n_ref)
    return res

  def embed(self, x: str) -> torch.Tensor:
    if self.input_mode == 'ch':
      ids = self.query_ids(x)
      rid = random.randint(0, len(ids)-1)
      ids = ids[rid]
      rid = f'{x}-{rid}'
    else:
      ids, rid = x, None

    tokens = [self.vocabulary_map['pad']] * self.max_len
    count = 0
    for s in ids:
      tokens[count] = self.vocabulary_map[s]
      count += 1
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)

    # mask = torch.zeros(self.max_len, device=self.device)
    # mask[count:] = -torch.inf
    return self.embedding(tokens), self.embedding2(tokens)

  def forward(self, sequence: tuple) -> tuple[torch.Tensor]:
    embed = []
    embed2 = []
    for data in sequence:
      e, e2 = self.embed(data)
      embed.append(e)
      embed2.append(e2)
    embed = torch.stack(embed, dim=0)  # (bs*num, max_len, n_embed)
    embed2 = torch.stack(embed2, dim=0)
    return embed, embed2


class QuantExtEncoder(nn.Module):

  def __init__(self, adapter: VQAdapter, c_out: int):

    super().__init__()
    self.adapter = adapter
    self.c_in = self.adapter.vqgan.quantize.e_dim
    self.c_out = c_out

  def _init_mobilenetv3(self, setting, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)):
    '''torchvision/models/mobilenetv3.py'''

    reduce_divider = 2
    width_mult = 1
    # dilation = 1
    # bneck_conf = partial(mobilenetv3.InvertedResidualConfig, width_mult=width_mult)
    # setting = [
    #   bneck_conf(16, 3, 16, 16, True, "RE", 1, 1),  # C1
    #   bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # C2
    #   bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
    #   bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
    #   bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
    #   bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
    #   bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
    #   bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
    #   bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
    #   bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    #   bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    # ]
    c_last = mobilenetv3.InvertedResidualConfig.adjust_channels(1024 // reduce_divider, width_mult)  # C5

    layers: list[nn.Module] = []

    # building first layer
    firstconv_output_channels = setting[0].input_channels
    layers.append(Conv2dNormActivation(self.c_in, firstconv_output_channels, kernel_size=3, stride=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))

    # building inverted residual blocks
    for cnf in setting:
      layers.append(mobilenetv3.InvertedResidual(cnf, norm_layer))

    # building last several layers
    lastconv_input_channels = setting[-1].out_channels
    # lastconv_output_channels = 6 * lastconv_input_channels
    layers.append(Conv2dNormActivation(lastconv_input_channels, self.c_out, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))

    layers = nn.Sequential(*layers)
    # self.avgpool = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Sequential(
    #   nn.Linear(c_out, c_last),
    #   nn.Hardswish(inplace=True),
    #   nn.Dropout(p=0.2, inplace=True),
    #   nn.Linear(c_last, num_classes),
    # )

    for m in layers.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
    return layers

  def _init_efficientv2(self, setting, norm_layer=partial(nn.BatchNorm2d, eps=1e-3), stochastic_depth_prob=0.2):
    '''torchvision/models/efficientnet.py'''

    # dropout = 0.3
    layers: list[nn.Module] = []

    # building first layer
    firstconv_output_channels = setting[0].input_channels
    layers.append(Conv2dNormActivation(self.c_in, firstconv_output_channels, kernel_size=3, stride=1, norm_layer=norm_layer, activation_layer=nn.SiLU))

    # building inverted residual blocks
    total_stage_blocks = sum(cnf.num_layers for cnf in setting)
    stage_block_id = 0
    for cnf in setting:
      stage: list[nn.Module] = []
      for _ in range(cnf.num_layers):
        # copy to avoid modifications. shallow copy is enough
        block_cnf = copy.copy(cnf)
        # overwrite info if not the first conv in the stage
        if stage:
          block_cnf.input_channels = block_cnf.out_channels
          block_cnf.stride = 1
        # adjust stochastic depth probability based on the depth of the stage block
        sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
        stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
        stage_block_id += 1
      layers.append(nn.Sequential(*stage))

    # building last several layers
    lastconv_input_channels = setting[-1].out_channels
    layers.append(Conv2dNormActivation(
      lastconv_input_channels,
      self.c_out,
      kernel_size=1,
      norm_layer=norm_layer,
      activation_layer=nn.SiLU,
    ))
    layers = nn.Sequential(*layers) # [bs, 256, 2, 2]

    for m in layers.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        # nn.init._no_grad_fill_(m.weight, 1e-5)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)
    return layers

  def _init_enc(self, dropout_p=0):
    ConvBlk = partial(ConvBlock, norm='in', pad_type='reflect', dropout=dropout_p)
    ResBlk = partial(ResBlock, norm='in', dropout=dropout_p)
    C = 32
    return nn.Sequential(
      ConvBlk(self.c_in, C),  # 16x16x32
      ConvBlk(C * 1, C * 2),  # , downsample=True
      ConvBlk(C * 2, C * 4),  # , downsample=True
      ResBlk(C * 4, C * 4),
      ResBlk(C * 4, C * 4),
      ResBlk(C * 4, C * 8),  # , downsample=True
      ResBlk(C * 8, self.c_out)  # 16x16x256
    )

  def set_device(self, device: torch.device):
    self.device = device
    self.adapter.set_device(device)

  def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, x.shape[-1])  # torch.Size([48, 256])
    x = self.adapter.lookup_quant(x)  # torch.Size([48, 4, 16, 16])
    return x
  
  def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
    weight = self.adapter.get_codebook()  # (n_embed, embedding_dim), weight.requires_grad=False
    x = x.softmax(dim=-1)
    x = torch.einsum('bij,jk->bik', x, weight)  # [bs n_embed, 4]
    x = rearrange(x, 'b (h w) c -> b c h w', h=int(x.shape[1]**0.5))
    return x


class StyleEncoder(QuantExtEncoder):

  def __init__(self, adapter: VQAdapter, c_out=None, l_ids=45, dropout_p=0):
    super().__init__(adapter, c_out)
    self.n_head = 8
    self.q_linear = nn.Linear(c_out, c_out, bias=False)
    self.kv_linear = nn.Linear(c_out, 2 * c_out, bias=False)
    self.c_proj = nn.Linear(c_out, c_out, bias=False)
    self.layer_norm = nn.LayerNorm(c_out, bias=False, eps=1e-6)
    self.wpe = nn.Embedding(l_ids, c_out)
    self.stem = self._init_enc(dropout_p=dropout_p)
    self.cl_head = nn.Sequential(
      nn.Linear(c_out, 1),  # [b, 2, 256, 1]
      nn.Flatten(-2, -1),  # [b, 2, 256]
      nn.LayerNorm(256),
      nn.SiLU(True),
      nn.Dropout(dropout_p, inplace=True),
    )
    self.cl_fc = nn.Linear(256, c_out)  # [b, 2, c_out]
    self.dropout_p = dropout_p if self.training else 0
    self.dropkey_mask = DropKeyMask(dropout_p, is_causal=False)
    self.ln_q = nn.LayerNorm(c_out // self.n_head)
    self.ln_k = nn.LayerNorm(c_out // self.n_head)

    for m in self.modules():
      if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
          torch.nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
  
    # self.stem = self._init_efficientv2((
    #     efficientnet.FusedMBConvConfig(1, 3, 1, 24, 24, 1),
    #     efficientnet.FusedMBConvConfig(1, 3, 1, 24, 48, 1),
    #     efficientnet.FusedMBConvConfig(1, 3, 1, 48, 64, 2),
    #     efficientnet.MBConvConfig(1, 3, 1, 64, 128, 2),
    #     efficientnet.MBConvConfig(1, 3, 1, 128, 160, 2),
    #     efficientnet.MBConvConfig(1, 3, 1, 160, 256, 4),
    #   ),
    #   partial(nn.InstanceNorm2d, affine=False)
    # )

  def _structure_style_aggregation(self, ids:torch.Tensor, feat:torch.Tensor) -> torch.Tensor:
    ids += self.wpe(torch.arange(0, ids.shape[1], dtype=torch.long, device=ids.device))
    q = self.q_linear(ids)  # [b*2, len, c]
    q = rearrange(q, 'b l (h c) -> b h l c', h=self.n_head)
    k, v = self.kv_linear(feat).split(self.c_out, dim=2)  # [b*2, 256*n//2, 256]
    k = rearrange(k, 'b hw (h c) -> b h hw c', h=self.n_head)
    v = rearrange(v, 'b hw (h c) -> b h hw c', h=self.n_head)
    q, k = self.ln_q(q), self.ln_k(k)

    mask = self.dropkey_mask(1, 1, q.shape[2], k.shape[2], device=ids.device)
    v = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    v = rearrange(v, 'b h l c -> b l (h c)')  # [b*2, len, c]
    return v

  def forward(self, indices: torch.Tensor, ids: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    # b: batchsize; n: num_refs
    x: torch.Tensor = self.encode_indices(indices)  # [b*n, 4, 16, 16]
    n_ref = indices.shape[1]  # n

    x = self.stem(x)  # [b*n, 256, 16, 16]
    x = rearrange(x, '(b n) c h w -> b n (h w) c', n=n_ref)  # [b, n, 256, c]
    sim *= 3
    # 根据相似度计算加权平均风格
    x_g = torch.einsum('b n f c, b n -> b f c', x, sim.softmax(dim=1))  # [b, 256, c]

    x = rearrange(x, 'b n f c -> b (n f) c')  # [b, 256*n, c]
    x = self._structure_style_aggregation(ids, x)  # [b, len, c]
    x_l = self.c_proj(x)  # [b, len, c]
    x = torch.concat((x_l, x_g), dim=1)  # [b, len+256, c]
    x_sss = self.layer_norm(x)  # [b, len+256, c]

    if not self.training:
      return x_sss, None

    x_cl = self.cl_fc(self.cl_head(x_g))  # [b, c]
    return x_sss, x_cl


class ContentEncoder(QuantExtEncoder):
  def __init__(self, adapter: VQAdapter, c_out=None):
    super().__init__(adapter, c_out)
    self.stem = self._init_efficientv2((
      efficientnet.FusedMBConvConfig(1, 3, 1, 24, 24, 2),
      efficientnet.FusedMBConvConfig(4, 3, 2, 24, 48, 2),
      efficientnet.FusedMBConvConfig(4, 3, 1, 48, 64, 4),
      efficientnet.MBConvConfig(4, 3, 2, 64, 128, 4),
      efficientnet.MBConvConfig(6, 3, 1, 128, 160, 4),
      efficientnet.MBConvConfig(6, 3, 2, 160, 256, 6),
    ))

  def forward(self, x: torch.Tensor):
    if len(x.shape) == 2:
      x = self.encode_indices(x)
    else:
      x = self.encode_logits(x)
    x = self.stem(x)  # [b, c_out, 2, 2]
    x = rearrange(x, 'b c h w -> b (h w) c')  # [b, 4, c_out]
    return x


class MoCoWrapper(nn.Module):
  def __init__(self, adapter: VQAdapter, c_out=None, l_ids=45, momentum=0.99):
    super().__init__()
    self.adapter = adapter
    self.momentum = momentum
    self.enc = StyleEncoder(adapter, c_out, l_ids, 0.1)
    self.enc_m = StyleEncoder(adapter, c_out, l_ids, 0.1)
    self._build_projector_and_predictor_mlps(c_out, 1024)
    self.enc_sync()
    self.enc_m.requires_grad_(False)

  def set_device(self, device: torch.device):
    self.device = device
    self.adapter.set_device(device)
    self.enc.set_device(device)
    self.enc_m.set_device(device)

  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
      dim1 = input_dim if l == 0 else mlp_dim
      dim2 = output_dim if l == num_layers - 1 else mlp_dim
      mlp.append(nn.Linear(dim1, dim2, bias=False))
      if l < num_layers - 1:
        mlp.append(nn.BatchNorm1d(dim2))
        mlp.append(nn.ReLU(inplace=True))
      elif last_bn:
        mlp.append(nn.BatchNorm1d(dim2, affine=False))
    # mlp.append(Rearrange('(b p) c -> b p c', p=2))
    return nn.Sequential(*mlp)

  def _build_projector_and_predictor_mlps(self, dim=256, mlp_dim=4096):
    hidden_dim = self.enc.cl_fc.weight.shape[1]
    del self.enc.cl_fc, self.enc_m.cl_fc
    self.enc.cl_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
    self.enc_m.cl_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
    self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
  
  def enc_sync(self):
    for param_q, param_k in zip(self.enc.parameters(), self.enc_m.parameters()):
      param_k.data.copy_(param_q.data)

  @torch.no_grad()
  def momentum_update(self, ratio):
    # https://github.com/AidenDurrant/MoCo-Pytorch/blob/master/src/model/moco.py#L72
    m = 1. - 0.5 * (1. + math.cos(math.pi * ratio)) * (1. - self.momentum)

    for param_q, param_k in zip(self.enc.parameters(), self.enc_m.parameters()):
      param_k.data = param_k.data * m + param_q.detach().data * (1. - m)

  def forward0(self, *args, **kwargs):
    sss, cl = self.enc(*args, **kwargs)
    if not self.training:
      return sss, None
    cl = self.predictor(cl)
    _, cl_m = self.enc_m(*args, **kwargs)
    cl_m = cl_m.detach()  # [b, 2, c]
    rand_i = random.randint(0, 1)
    cl = torch.stack((cl[:, rand_i], cl_m[:, 1-rand_i]), dim=1)
    return sss, (cl, cl_m, )

  def forward(self, *args, **kwargs):
    sss, cl = self.enc(*args, **kwargs)
    if not self.training:
      return sss, None
    cl = self.predictor(cl)
    _, cl_m = self.enc_m(*args, **kwargs)
    cl_m = cl_m.detach()  # [b, c]
    cl = torch.stack((cl, cl_m), dim=1)  # [b, 2, c]
    return sss, cl
