import os
import sys
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFont
import lightning.pytorch as pl

from data import cn
from data import valid_characters
from util import utils
from util.io import write_json
from data.adapter import pil_to_tensor


def get_radical_dict():
  radicals = {}
  path = cn.RAW_FILE_DIR / 'IDS_Dictionary.txt'
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      ch, comps = line.split(':')
      radicals[ch] = [c for c in comps if ord(c) > 1e4 and c not in cn.IDC]
  return radicals


class IFFontDataset(Dataset):
  
  def __init__(self, ttf_root=None, corpus='', img_size=128, mode='L', ch_num=None, font_num=None):
    self.corpus = corpus
    self.img_size = img_size
    self.img_mode = mode
    self.ch_size = ch_num or len(self.corpus)

    self.fonts_path = []
    for f in os.listdir(ttf_root):
      path = os.path.join(ttf_root, f)
      if not os.path.isfile(path):
        continue
      self.fonts_path.append(path)

    num = len(self.fonts_path)
    font_num = num if font_num is None else min(num, font_num)
    self.fonts_path = self.fonts_path[::num//(font_num or num)][:font_num]
    self.font_size = font_num

    self.fonts = [ImageFont.truetype(f, self.img_size) for f in self.fonts_path]
    self.split = os.path.split(ttf_root)[0]
    self.radical_dict = get_radical_dict()

  def __len__(self):
    return self.font_size * self.ch_size

  def _synthesis_img(self, ch, font):
    img = utils.draw_single_char(ch, font, self.img_size, mode=self.img_mode)
    # torch.Size([channel, 128, 128])
    # img = np.array(img)
    warnings.filterwarnings("ignore", category=UserWarning)
    return pil_to_tensor(img)
  
  def _get_one(self, ch_id, font_id, name):
    ch = self.corpus[ch_id]
    font = self.fonts[font_id]
    img = self._synthesis_img(ch, font)
    # img = F.pil_to_tensor(img.convert("RGB"))
    # print('dataset', img.max(), img.min())
    radical = self.radical_dict[ch]
    return {
      name: img,
      f'{name}_writerId': font_id,
      f'{name}_label': ch,
      f'{name}_lexicon': radical,
    }

  def write_fonts_mappging(self, path, suffix):
    try:
      write_json(
        os.path.join(path, f'fonts_mapping_{suffix}.json'), 
        {i: os.path.split(v)[1] for i, v in enumerate(self.fonts_path)},
        indent=2,
      )
    except OSError as e:
      print(e, file=sys.stderr)
  
  @staticmethod
  def collate(data_list):
    res = {}
    for d in data_list:
      for k, v in d.items():
        res.setdefault(k, [])
        res[k].append(v)
    for k, v in res.items():
      if isinstance(v[0], torch.Tensor):
        res[k] = torch.stack(v, dim=0)
        continue
      # res[k] = torch.as_tensor(v)
    return res


class ValDataset(IFFontDataset):
  def __init__(self, ttf_root=None, corpus='', img_size=128, mode='L', ch_num=10, font_num=6):
    super().__init__(ttf_root, corpus, img_size, mode, ch_num, font_num)


class TestDataset(IFFontDataset):
  def __init__(self, ttf_root=None, corpus='', img_size=128, mode='L', ch_num=None, font_num=None):
    super().__init__(ttf_root, corpus, img_size, mode, ch_num, font_num)
    self._check_corpus()

  def _check_corpus(self):
    ch_sorted = {'seen': [], 'unseen': [], 'unknown': [],}
    for ch in self.corpus:
      if ch in valid_characters.train_ch:
        ch_sorted['seen'].append(ch)
      elif ch in valid_characters.val_ch:
        ch_sorted['unseen'].append(ch)
      else:
        ch_sorted['unknown'].append(ch)

    for k, v in ch_sorted.items():
      print(f'{k}_ch: {"".join(v)}')


def init_dataset(ttf_root, split='train', **kwargs):
  type_ = split.split('_', 1)[0]
  ch = kwargs.pop('corpus', None) or getattr(valid_characters, f'{type_}_ch')
  ch_num = len(ch)
  ch_num = min(ch_num, kwargs.pop('ch_num', None) or ch_num)
  font_num = kwargs.pop('font_num', None)

  dataset = IFFontDataset
  if type_ == 'val':
    dataset = ValDataset
  elif type_ == 'test':
    dataset = TestDataset

  dataset = dataset(ttf_root, ch, img_size=kwargs.pop('size'), mode=kwargs.pop('mode'), ch_num=ch_num, font_num=font_num)
  ckp_path = kwargs.pop('checkpoint_path', None)
  if ckp_path:
    dataset.write_fonts_mappging(ckp_path, split)
  return dataset


class DataModuleFromConfig(pl.LightningDataModule):
  def __init__(self, logdir, img_size, img_mode, data_dir, batch_size=1, val_batch_size=2, num_workers=1, pin_memory=False):
    super().__init__()
    self.batch_size = batch_size
    self.val_batch_size = val_batch_size
    self.dset_kwargs = {
      'size': img_size,
      'mode': img_mode,
      'checkpoint_path': logdir,
    }
    self.loader_kwargs = {
      'num_workers': num_workers,
      'pin_memory': pin_memory,
    }
    self.data_dir = data_dir

  def prepare_data(self):
    ...

  def setup(self, stage: str):
    if stage == 'fit':
      self.dset_train =  init_dataset(
        os.path.join(self.data_dir, 'train'),
        split='train',
        ch_num=None,
        font_num=None,
        **self.dset_kwargs,
      )
      self.dset_val = init_dataset(
        os.path.join(self.data_dir, 'train'),
        split='val_seen',
        ch_num=None,
        font_num=None,
        **self.dset_kwargs,
      )
    elif stage == 'test':
      self.dset_test = init_dataset(
        os.path.join(self.data_dir, 'val'),
        split='val_unseen',
        ch_num=2,
        font_num=2,
        **self.dset_kwargs,
      )

  def train_dataloader(self):
    d = self.dset_train
    return DataLoader(d, batch_size=self.batch_size, shuffle=True, collate_fn=d.collate, **self.loader_kwargs)

  def val_dataloader(self):
    d = self.dset_val
    return DataLoader(d, batch_size=self.val_batch_size, shuffle=False, collate_fn=d.collate, **self.loader_kwargs)

  def test_dataloader(self):
    d = self.dset_test
    return DataLoader(d, batch_size=self.val_batch_size, shuffle=False, collate_fn=d.collate, **self.loader_kwargs)
  
  def teardown(self, stage: str):
    ...
