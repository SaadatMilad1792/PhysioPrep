########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import numpy as np
import physioprep as pp

########################################################################################################################
## -- tests for validation modules -- ##################################################################################
########################################################################################################################
def test_validation_module():
  validator = pp.M3WaveFormValidationModule()

  arr_clean = np.array([[1, 2], [3, 4]])
  assert validator.is_not_nan(arr_clean) is True
  assert validator.apply(arr_clean) is True

  arr_nan = np.array([[1, np.nan], [3, 4]])
  assert validator.is_not_nan(arr_nan) is False
  assert validator.apply(arr_nan) is False

  arr_all_nan = np.full((2, 2), np.nan)
  assert validator.is_not_nan(arr_all_nan) is False
  assert validator.apply(arr_all_nan) is False

  arr_empty = np.array([])
  assert validator.is_not_nan(arr_empty) is True
  assert validator.apply(arr_empty) is True
