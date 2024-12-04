from functools import partial

import numpy as np
import torch
import torchmetrics
from torchmetrics import regression as tm_regress
from torchmetrics import image as tm_image


class RootMeanSquaredError(tm_regress.MeanSquaredError):
  def compute(self) -> torch.Tensor:
    return super().compute().sqrt()


class MetricWrapper:
  METRICS_SUPPORTED = dict(
    L1=tm_regress.MeanAbsoluteError,
    MSE=tm_regress.MeanSquaredError,
    RMSE=RootMeanSquaredError,
    SSIM=partial(tm_image.StructuralSimilarityIndexMeasure, data_range=(0, 1)),
    LPIPS=partial(tm_image.LearnedPerceptualImagePatchSimilarity, net_type='squeeze'),  # pip install lpips
    FID=partial(tm_image.FrechetInceptionDistance, feature=2048),  # pip install torch-fidelity
    PSNR=partial(tm_image.PeakSignalNoiseRatio, data_range=(0, 1)),
  )

  def __init__(self, *metrics_enable:str, data_range:tuple[int]=(0, 255)) -> None:
    assert isinstance(data_range, tuple) and data_range[1] > data_range[0]
    self.d_range = data_range
    self.device = None
    self.METRICS_ENABLE = tuple(map(lambda m: m.upper(), metrics_enable)) or tuple(self.METRICS_SUPPORTED.keys())
    for m_name in self.METRICS_ENABLE:
      m_cls = self.METRICS_SUPPORTED.get(m_name)
      assert m_cls is not None, f'metric {m_name} is not supported!'
      setattr(self, m_name, m_cls())

  def _iter_metrics(self):
    for k, v in vars(self).items():
      if not isinstance(v, torchmetrics.Metric):
        continue
      yield (k, v, )

  def set_device(self, device:torch.device):
    for name, obj in self._iter_metrics():
      setattr(self, name, obj.to(device))
    self.device = device

  def _normalize(self, x:torch.Tensor):
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x)
    low, high = self.d_range[:2]
    x = torch.clamp(x, low, high)
    x = (x - low) / (high - low)
    if self.device is not None:
      x = x.to(self.device)
    return x

  @torch.no_grad()
  def step(self, x:torch.Tensor, y:torch.Tensor, mode:str='update') -> dict[str, torch.Tensor|None]:
    assert mode in {'update', 'forward'}
    x = self._normalize(x)
    y = self._normalize(y)
    res = dict()
    ZERO_TENSOR = torch.as_tensor(0, dtype=torch.float, device=x.device)

    for name, obj in self._iter_metrics():
      method = getattr(obj, mode)
      if isinstance(obj, tm_image.LearnedPerceptualImagePatchSimilarity):
        val = method(x*2-1, y*2-1)
      elif isinstance(obj, tm_image.FrechetInceptionDistance):
        obj.update((x * 255).type(torch.uint8), real=False)
        obj.update((y * 255).type(torch.uint8), real=True)
        val = ZERO_TENSOR
      else:
        val = method(x, y)
      res[name] = val or ZERO_TENSOR
    
    return res

  @torch.no_grad()
  def compute(self) -> dict[str, torch.Tensor]:
    res = dict()
    for name, obj in self._iter_metrics():
      res[name] = obj.compute()
    return res

  def reset(self):
    for _, obj in self._iter_metrics():
      obj.reset()
