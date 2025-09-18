########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import numpy as np
import physioprep as pp

########################################################################################################################
## -- tests for validation modules -- ##################################################################################
########################################################################################################################
def test_is_not_nan_true_false():
  validator = pp.M3WaveFormValidationModule()
  arr_good = np.ones((2, 5))
  arr_bad = np.array([[1, 2, np.nan, 4, 5]])
  assert validator.is_not_nan(arr_good) is True
  assert validator.is_not_nan(arr_bad) is False

def test_has_sufficient_variance_pass():
  validator = pp.M3WaveFormValidationModule()
  arr = np.array([[1, 2, 3, 4, 5],
                  [5, 4, 3, 2, 1]])
  assert validator.has_sufficient_variance(arr, threshold=0.2) is True

def test_has_sufficient_variance_flat_under_threshold():
  validator = pp.M3WaveFormValidationModule()
  arr = np.array([[1, 1, 2, 3, 4, 5, 6, 7, 8, 9]])  # flat run length = 1
  assert validator.has_sufficient_variance(arr, threshold=0.3) is True

def test_has_sufficient_variance_flat_over_threshold():
  validator = pp.M3WaveFormValidationModule()
  arr = np.array([[1, 1, 1, 1, 2, 3, 4, 5, 6, 7]])  # flat run length = 3
  assert validator.has_sufficient_variance(arr, threshold=0.2) is False

def test_has_sufficient_variance_no_flats():
  validator = pp.M3WaveFormValidationModule()
  arr = np.array([[1, 2, 3, 4, 5]])
  assert validator.has_sufficient_variance(arr, threshold=0.2) is True

def test_apply_integration():
  validator = pp.M3WaveFormValidationModule()
  good = np.arange(20).reshape(1, -1)
  bad_nan = np.array([[1, 2, np.nan, 4]])
  bad_flat = np.array([[1] * 20])

  assert validator.apply(good) is True
  assert validator.apply(bad_nan) is False
  assert validator.apply(bad_flat) is False

def test_has_sufficient_variance_min_flat_len():
  validator = pp.M3WaveFormValidationModule()
  arr = np.array([[1]])
  assert validator.has_sufficient_variance(arr, threshold=0.01) is True
