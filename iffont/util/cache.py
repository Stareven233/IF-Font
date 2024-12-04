class CacheManager:
  def __init__(self, size, *keys:str) -> None:
    self.cache_map: dict[str, CacheQueue] = dict()
    for k in keys:
      self.cache_map[k] = CacheQueue(size)

  def push(self, *values):
    for k, v in zip(self.cache_map.keys(), values):
      self.cache_map[k].push(v)
  
  def pop(self, size=None):
    ret = []
    for cache in self.cache_map.values():
      r = cache.pop(size)
      ret.append(r)
    return ret

  def reset(self):
    for cache in self.cache_map.values():
      cache.reset()


class CacheQueue:
  def __init__(self, size:int) -> None:
    self.data = []
    self.size = size
    self.last = -1
    self.next = 0
    self.is_full = False
  
  def push(self, value):
    if self.is_full:
      self.data[self.next] = value
    else:
      self.data.append(value)
    self.last = self.next
    self.next = (self.next + 1) % self.size
    self.is_full = self.is_full or (self.next == 0)

  def pop(self, size=None):
    ret = self.data[:size]
    return ret
  
  def reset(self):
    self.data.clear()
    self.last = -1
    self.next = 0
    self.is_full = False
