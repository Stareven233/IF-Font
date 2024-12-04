from pathlib import Path
import re
from collections.abc import Callable
import itertools

from util.io import load_json, write_json
from util.utils import ROOT_DIR


# Ideographic Description Characters, U+2FF0 to U+2FFB
IDC = ('⿰','⿱','⿲','⿳','⿴','⿵','⿶','⿷','⿸','⿹','⿺','⿻', )
N_IDC_COMPS = {'⿰': 2, '⿱': 2, '⿲': 3, '⿳': 3, '⿴': 2, '⿵': 2, '⿶': 2, '⿷': 2, '⿸': 2, '⿹': 2, '⿺': 2, '⿻': 2}
IDS_VOCABULARY = '⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻㇌㇏丶㇊𠃊丨㇉乛㇅𡿨𠃌㇇一𠃋𠄎㇎⺄乙㇋𠄌㇄𠃍㇁㇒㇀㇂乚亅丿'
RAW_FILE_DIR = ROOT_DIR / '..' / 'data' / 'raw_files'

# 3500常用字表 from: https://github.com/mapull/chinese-dictionary/blob/main/data/character/common/char_common_base.json
__data = load_json(RAW_FILE_DIR / 'chars.json')

'''500个最常用字'''
CN500 = __data['cn500']

'''2500个常用汉字'''
CN2500 = __data['cn2500']

'''1000个次常用汉字'''
CN1000 = __data['cn1000']

'''3500个常用汉字'''
CN3500 = __data['cn3500']

CN5000 = __data['cn5000']

'''
  常用6763个汉字使用频率表

  原文地址：http://blog.sina.com.cn/s/blog_5e2ffb490100dnfg.html

  汉字频度表统计资料来源于清华大学，现公布如下，仅供参考。
      使用字数   6763   字（国标字符集），范文合计总字数   86405823 个。
      说明如下：

      假若认识  500 字，则覆盖面为  78.53 % 。其余类推，

  列表如下：
  字数          覆盖面（  % ）
    500        78.53202
  1000        91.91527
  1500        96.47563
  2000        98.38765
  2500        99.24388
  3000        99.63322
  3500        99.82015
  4000        99.91645
  4500        99.96471
  5000        99.98633
  5500        99.99553
  6000        99.99901
  6479       100.00000
  6500       100.00000
  6763       100.00000 
'''
FREQ_CHARACTERS = __data['freq_characters']
FREQ_COMPONENTS = __data['freq_components']


'''每个笔画letter所代表的详细信息, 26种笔画'''
stroke_detail = load_json(RAW_FILE_DIR / 'stroke-table.json')
'''汉字到对应笔画letter序列的映射, 6939个字'''
stroke_map = load_json(RAW_FILE_DIR / 'stroke-order-jian.json')


def resolve_IDS_Dictionary(ch_set, save_path=None, recursive=True, only_idc=False) -> tuple[dict[str, str], set[str]]:
  # 〇 代表独体字
  IDC = '⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻〇'

  __data = dict()
  with open(RAW_FILE_DIR / 'IDS_Dictionary.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      ch, comps = line.split(':')
      __data[ch] = comps.strip(' \n').split(' ')

  def _search_ids(ch) -> list[str]:
    res = []
    comps: list = __data.get(ch, [ch])
    if not recursive:
      res = comps
      comps = tuple()
    for c in comps:
      if c in IDC or c not in __data or len(__data[c]) == 1:
        res.append(c)
      else:
        res.extend(_search_ids(c))
    if only_idc:
      res = list(filter(lambda x: x in IDC, res))
      if len(res) == 0:
        res.append(IDC[-1])
    return res

  ids_map = dict()
  ids_set = set()
  for ch in ch_set:
    t = _search_ids(ch)
    ids_set.update(t)
    ids_map[ch] = ''.join(t)

  if save_path is not None:
    save_path = Path(save_path)
    write_json(save_path / 'ids_map.json', ids_map)
    write_json(save_path / 'ids_set.json', tuple(ids_set))
  return ids_map, ids_set


def resolve_IDS_babelstone(characters:str|None=None, level='stroke') -> tuple[dict, set, dict]:
  pattern_comp = re.compile(r'#\t(\{\d+\})\t.*?\t(.*)')
  pattern_ch = re.compile(r'U\+[0-9A-Z]+\t(.).*?\t\^(.+?)\$')
  
  pattern_split_ids = re.compile(r'(?<![0-9])(?![0-9])')
  ch_empty, ch_unencoded = dict(), set()

  def read_ids(p:Path, match:Callable) -> dict:
    d = dict()
    f = p.open('r', encoding='utf-8')
    while (line := f.readline()):
      m = match(line)
      if m is None:
        continue
      d[m.group(1)] = m.group(2)
    f.close()
    return d
  
  def resolve_stroke(c:str) -> str:
    if c not in ch_unencoded and re.match(r'\{\d+\}', c):
      ch_unencoded.add(c)
    ids = ids_map.get(c)
    if ids is None:
      ch_empty.setdefault(c, [])
      ch_empty[c].append(ch)
      ids = c

    if c in ch_resolved:
      return ids
    if c == ids:
      ch_base.setdefault(c, [])
      ch_base[c].append(ch)
      ch_resolved.add(c)
    else:
      ids = filter(lambda x: len(x)>0, pattern_split_ids.split(ids))
      ids = itertools.chain(*map(resolve_stroke, ids))
      ch_resolved.add(c)

    ids_map[c] = tuple(ids)
    return ids_map[c]

  def resolve_radical(c:str) -> tuple[str]:
    if c not in ch_unencoded and re.match(r'\{\d+\}', c):
      ch_unencoded.add(c)
    ids = ids_map.get(c)

    if ids is None or ids == '':
      ch_empty.setdefault(c, [])
      ch_empty[c].append(ch)
      ids = ids_map[c] = (c, )
    if c in ch_resolved:
      return ids

    is_radical = True
    ids_filtered = []
    for i in pattern_split_ids.split(ids):
      if len(i) == 0:
        continue
      ids_filtered.append(i)
      is_radical &= len(ids_map_origin.get(i, '')) <= 1
    is_radical |= (c == ids)
    is_radical |= (ids_filtered[0] == '⿻')
  
    if is_radical:
      ch_base.setdefault(c, [])
      ch_base[c].append(ch)
      ids_map[c] = (c, )
      ch_resolved.add(c)
      return ids_map[c]
  
    ids = itertools.chain(*map(resolve_radical, ids_filtered))
    ids_map[c] = tuple(ids)
    ch_resolved.add(c)
    return ids_map[c]

  ids_map, ch_resolved, ch_base = dict(), set(IDC), dict()
  for i in IDC:
    ids_map[i] = i
  ids_map |= read_ids(
    RAW_FILE_DIR / 'babelstone.co.uk_CJK_IDS.TXT', 
    lambda x: pattern_ch.match(x) or pattern_comp.match(x),
  )

  pattern_ch = re.compile(r'^([^#]+)\t([^#\s]+)')
  ids_supplement = read_ids(RAW_FILE_DIR / 'ids_iffont.txt', lambda x: pattern_ch.match(x))
  ids_map |= ids_supplement
  ids_map_origin = ids_map.copy()

  resolve_method = locals()[f'resolve_{level}']
  ids_max_len = 0
  for ch in (characters or tuple(ids_map.keys())):
    resolve_method(ch)
    ids_max_len = max(ids_max_len, len(ids_map[ch]))

  return ids_map, ch_resolved, ch_base


del __data
