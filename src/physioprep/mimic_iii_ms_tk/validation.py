########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import numpy as np

########################################################################################################################
## -- validation modules -- ############################################################################################
########################################################################################################################
class M3WaveFormValidationModule:
  def __init__(self):
    super(M3WaveFormValidationModule, self).__init__()

  def is_not_nan(self, arr: np.ndarray) -> bool:
    return not np.isnan(arr).any()

  def has_sufficient_variance(self, arr: np.ndarray, threshold: float = 0.1) -> bool:
    for channel in arr:
      max_flat_len = int(len(channel) * threshold)
      if max_flat_len < 1:
        max_flat_len = 1

      count = 1
      for i in range(1, len(channel)):
        if channel[i] == channel[i-1]:
          count += 1
          if count >= max_flat_len:
            return False
        else:
          count = 1
    return True

  def apply(self, arr: np.ndarray) -> bool:
    flag = self.is_not_nan(arr) and \
           self.has_sufficient_variance(arr)
           
    return flag