########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import pytest
import pandas as pd
import physioprep as pp
import requests
from unittest.mock import patch, MagicMock

########################################################################################################################
## -- tests for utility functions -- ###################################################################################
########################################################################################################################

## -- tests for get subject split -- ##
@pytest.fixture
def sample_split_df():
  return pd.DataFrame({
    'patient_id': list(range(10)),
    'value': list(range(10))
  })

def test_single_split(sample_split_df):
  splits = pp.get_subject_split(sample_split_df, seed = 42)
  assert len(splits) == 1
  assert set(splits[0]['patient_id']) == set(sample_split_df['patient_id'])
  assert len(splits[0]) == len(sample_split_df)

def test_two_splits(sample_split_df):
  splits = pp.get_subject_split(sample_split_df, split_ratio = [0.7, 0.3], seed = 42)
  assert len(splits) == 2
  total_rows = sum(len(df) for df in splits)
  assert total_rows == len(sample_split_df)
  splits_again = pp.get_subject_split(sample_split_df, split_ratio = [0.7, 0.3], seed = 42)
  for s1, s2 in zip(splits, splits_again):
    assert set(s1['patient_id']) == set(s2['patient_id'])

def test_multiple_splits(sample_split_df):
  splits = pp.get_subject_split(sample_split_df, split_ratio = [1, 2, 7], seed = 123)
  assert len(splits) == 3
  total_rows = sum(len(df) for df in splits)
  assert total_rows == len(sample_split_df)

def test_empty_dataframe():
  empty_df = pd.DataFrame(columns=['patient_id', 'value'])
  splits = pp.get_subject_split(empty_df, split_ratio = [0.5, 0.5])
  assert len(splits) == 2
  assert all(df.empty for df in splits)

## -- tests for filter patient waveforms -- ##
@pytest.fixture
def sample_filter_df():
  return pd.DataFrame({
    'patient_id': [1, 2, 3],
    'signals': [
      ['ECG', 'PPG'],
      ['ECG'],
      ['PPG', 'SpO2', 'ECG']
    ]
  })

def test_filter_all_mode(sample_filter_df):
  channels = ['ECG', 'PPG']
  filtered = pp.filter_patient_waveforms(sample_filter_df, channels, mode = "all")
  assert set(filtered['patient_id']) == {1, 3}

def test_filter_any_mode(sample_filter_df):
  channels = ['PPG', 'SpO2']
  filtered = pp.filter_patient_waveforms(sample_filter_df, channels, mode = "any")
  assert set(filtered['patient_id']) == {1, 3}

def test_filter_no_match(sample_filter_df):
  channels = ['EEG']
  filtered = pp.filter_patient_waveforms(sample_filter_df, channels, mode = "all")
  assert filtered.empty

def test_filter_invalid_mode(sample_filter_df):
  channels = ['ECG']
  with pytest.raises(ValueError) as excinfo:
    pp.filter_patient_waveforms(sample_filter_df, channels, mode = "unknown")
  assert "Unknown mode" in str(excinfo.value)

## -- tests for fetch with retry -- ##
def test_success_first_try():
  mock_func = MagicMock(return_value = "ok")
  mock_func.__name__ = "success_first_try"
  result = pp.fetch_with_retry(mock_func)
  assert result == "ok"
  mock_func.assert_called_once()

def test_success_after_retry():
  mock_func = MagicMock(side_effect = [requests.ConnectionError, "ok"])
  mock_func.__name__ = "success_after_retry"
  with patch("time.sleep", return_value = None) as mock_sleep:
    result = pp.fetch_with_retry(mock_func, retries = 3, delay = 1)
  assert result == "ok"
  assert mock_func.call_count == 2
  mock_sleep.assert_called_once()

def test_timeout_then_success():
  mock_func = MagicMock(side_effect = [requests.Timeout, "fine"])
  mock_func.__name__ = "timeout_then_success"
  with patch("time.sleep", return_value = None):
    result = pp.fetch_with_retry(mock_func, retries = 2, delay = 1)
  assert result == "fine"
  assert mock_func.call_count == 2

def test_fail_all_retries():
  mock_func = MagicMock(side_effect = requests.ConnectionError)
  mock_func.__name__ = "fail_all_retries"
  with patch("time.sleep", return_value = None) as mock_sleep:
    with pytest.raises(Exception) as excinfo:
      pp.fetch_with_retry(mock_func, retries = 3, delay = 1)
  assert "Failed after 3 retries" in str(excinfo.value)
  assert mock_func.call_count == 3
  assert mock_sleep.call_count == 3

def test_fail_with_timeout_all_retries():
  mock_func = MagicMock(side_effect = requests.Timeout)
  mock_func.__name__ = "fail_with_timeout_all_retries"
  with patch("time.sleep", return_value = None):
    with pytest.raises(Exception) as excinfo:
      pp.fetch_with_retry(mock_func, retries = 2, delay = 1)
  assert "Failed after 2 retries" in str(excinfo.value)
  assert mock_func.call_count == 2
