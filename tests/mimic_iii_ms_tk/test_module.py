########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import wfdb
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import physioprep as pp
from concurrent.futures import Future
from physioprep import M3WaveFormMasterClass

########################################################################################################################
## -- tests for physioprep/mimic_iii_ms_tk/module.py -- ################################################################
########################################################################################################################

## -- tests for preset metadata data frame -- ##
def test_get_preset_returns_dataframe():
  module = pp.M3WaveFormMasterClass()
  result = module.get_preset()
  assert isinstance(result, pd.DataFrame)

## -- tests for get patients -- ##
def test_get_patients():
  obj = pp.M3WaveFormMasterClass()
  obj.args = {
    "physionet_url": "http://example.com/",
    "physionet_dir": "data/"
  }

  mock_response = MagicMock()
  mock_response.text = "patient1\npatient2\npatient3"

  with patch("requests.get", return_value = mock_response) as mock_get:
    patients = obj.get_patients()

  mock_get.assert_called_once_with("http://example.com/data/RECORDS")
  assert isinstance(patients, list)
  assert patients == ["patient1", "patient2", "patient3"]
  assert all(isinstance(p, str) for p in patients)

## -- tests for extracting patient groups and ids -- ##
def test_get_patient_group_id():
  obj = pp.M3WaveFormMasterClass()

  patient_group_id = "groupA/pid123/"
  group, pid = obj.get_patient_group_id(patient_group_id = patient_group_id)
  assert group == "groupA"
  assert pid == "pid123"

  patient_group_id = "groupB/pid999/"
  group, pid = obj.get_patient_group_id(patient_group_id = patient_group_id)
  assert group == "groupB"
  assert pid == "pid999"

## -- tests for getting patient records -- ##
def test_get_records():
  obj = pp.M3WaveFormMasterClass()
  obj.args = {
    "physionet_url": "http://example.com/",
    "physionet_dir": "data/"
  }

  mock_response = MagicMock()
  mock_response.text = "p123456-1234-12-12-12-12\ninvalid_record\np654321-4321-01-01-01-01"
  
  with patch("requests.get", return_value = mock_response) as mock_get:
    records = obj.get_records(patient_group_id = "groupA/pid123/")

  mock_get.assert_called_once_with("http://example.com/data/groupA/pid123/RECORDS")
  assert isinstance(records, list)
  assert records == ["p123456-1234-12-12-12-12", "p654321-4321-01-01-01-01"]
  assert all(isinstance(r, str) for r in records)

## -- tests for extracting patient headers -- ##
def test_get_patient_header():
  obj = pp.M3WaveFormMasterClass()
  obj.args = {
    "physionet_url": "http://example.com/",
    "physionet_dir": "data/"
  }

  patient_group_id = "groupA/pid123/"
  record = "p123456-1234-12-12-12-12"
  mock_header = MagicMock(spec = wfdb.Record)
  with patch("wfdb.rdheader", return_value = mock_header) as mock_rdheader:
    group, pid = obj.get_patient_group_id(patient_group_id)
    header = obj.get_patient_header(group = group, pid = pid, record = record)

  mock_rdheader.assert_called_once_with(record, pn_dir = "data/groupA/pid123/")
  assert header is mock_header

## -- tests for extracting segments from a record -- ##
def test_get_record_segments():
  obj = pp.M3WaveFormMasterClass()
  obj.args = {
    "physionet_url": "http://example.com/",
    "physionet_dir": "data/"
  }

  patient_group_id = "groupA/pid123/"
  record = "p123456-1234-12-12-12-12"
  mock_response = MagicMock()
  mock_response.text = (
    "1234567_0001 info1\n"
    "invalid_line\n"
    "7654321_0002 info2"
  )

  with patch("requests.get", return_value = mock_response) as mock_get:
    record_ids, record_inf = obj.get_record_segments(patient_group_id = patient_group_id, record = record)

  mock_get.assert_called_once_with("http://example.com/data/groupA/pid123/p123456-1234-12-12-12-12.hea")
  assert record_ids == ["1234567_0001", "7654321_0002"]
  assert record_inf == ["info1", "info2"]
  assert all(isinstance(r, str) for r in record_ids)
  assert all(isinstance(i, str) for i in record_inf)

## -- testing certain flags -- ##
def test_contains_certain():
  obj = pp.M3WaveFormMasterClass()
  obj.args = {
    "physionet_url": "http://example.com/",
    "physionet_dir": "data/"
  }

  patient_group_id = "groupA/pid123/"
  record_segment = "1234567_0001"
  mock_response = MagicMock()
  mock_response.text = "normal line\nanother line"
  with patch("requests.get", return_value = mock_response) as mock_get:
    result = obj.contains_certain(patient_group_id = patient_group_id, record_segment = record_segment)

  mock_get.assert_called_once_with("http://example.com/data/groupA/pid123/1234567_0001.hea")
  assert result is True

  mock_response2 = MagicMock()
  mock_response2.text = "normal line\ncertain alignment\nanother line"
  with patch("requests.get", return_value = mock_response2):
    result2 = obj.contains_certain(patient_group_id = patient_group_id, record_segment = record_segment)

  assert result2 is True

## -- tests for extracting signals within a segment -- ##
def test_get_signals_within():
  obj = pp.M3WaveFormMasterClass()
  patient_group_id = "groupA/pid123/"
  record_segment = "1234567_0001"
  mock_header = MagicMock()
  mock_header.sig_name = ["ECG", "PPG", "RESP"]
  obj.get_patient_header = MagicMock(return_value = mock_header)
  signals = obj.get_signals_within(patient_group_id = patient_group_id, record_segment = record_segment)
  group, pid = obj.get_patient_group_id(patient_group_id)
  obj.get_patient_header.assert_called_once_with(group, pid, record_segment)
  assert isinstance(signals, list)
  assert signals == ["ECG", "PPG", "RESP"]
  assert all(isinstance(s, str) for s in signals)

## -- tests for create preset lookup -- ##
@pytest.fixture
def mock_obj():
  obj = pp.M3WaveFormMasterClass()
  obj.get_patients = MagicMock(return_value = ['p1', 'p2'])
  obj.get_records = MagicMock(return_value = ['r1', 'r2'])
  obj.get_record_segments = MagicMock(return_value = (['s1', 's2'], [10, 20]))
  obj.get_signals_within = MagicMock(return_value = ['ECG', 'PPG'])
  obj.contains_certain = MagicMock(return_value = True)
  obj.get_patient_group_id = MagicMock(return_value = ('G1', 'p1'))
  return obj

def test_create_preset_lookup_sequential(mock_obj):
  with patch("physioprep.mimic_iii_ms_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients = ['p1'], cores = None)
    assert not df.empty
    expected_cols = ['patient_group', 'patient_id', 'record', 'segment', 'certain', 'segment_len', 'signals']
    assert all(col in df.columns for col in expected_cols)

def test_create_preset_lookup_with_save(mock_obj, tmp_path):
  save_file = tmp_path / "lookup.pkl"
  with patch("physioprep.mimic_iii_ms_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients=['p1'], cores = 1, save_as = str(save_file))
    assert save_file.exists()
    expected_cols = ['patient_group', 'patient_id', 'record', 'segment', 'certain', 'segment_len', 'signals']
    assert all(col in df.columns for col in expected_cols)

def test_create_preset_lookup_threaded(mock_obj):
  mod = pp.M3WaveFormMasterClass()
  df = mod.create_preset_lookup(patients = ['p00/p000020/', 'p00/p000030/'], cores = 2)
  assert not df.empty

def test_create_preset_lookup_empty_result(mock_obj):
  mock_obj.get_records = MagicMock(return_value=[])
  with patch("physioprep.mimic_iii_ms_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients = ['p1'])
    assert df.empty


@pytest.fixture
def dummy_class():
  cls = pp.M3WaveFormMasterClass()
  return cls

def make_df():
  return pd.DataFrame({"x": [1]})

@patch("physioprep.mimic_iii_ms_tk.module.fetch_with_retry")
def test_full_coverage_single_core(mock_fetch, tmp_path, dummy_class):
  # mock fetch_with_retry behavior depending on call
  def side_effect(func, *args, **kwargs):
    if func.__name__ == "get_signals_within":
      return [0.1, 0.2]
    if func.__name__ == "contains_certain":
      return True
    if func.__name__ == "get_patient_group_id":
      return ("grp", "pid")
    if func.__name__ == "get_record_segments":
      return (["seg1", "seg2"], [10, 20])
    if func.__name__ == "get_records":
      return ["rec1", "rec2"]
    if func.__name__ == "get_patients":
      return ["pat1", "pat2"]
    raise ValueError("unexpected call")
  mock_fetch.side_effect = side_effect

  # test _process_segment_for_lookup
  df = dummy_class._process_segment_for_lookup("pat1", "rec1", "seg1", 10)
  assert isinstance(df, pd.DataFrame)
  assert "signals" in df.columns

  # test _process_record_for_lookup
  df_list = dummy_class._process_record_for_lookup("pat1", "rec1", [False, False, False])
  assert all(isinstance(x, pd.DataFrame) for x in df_list)

  # test _process_patient_for_lookup
  df_list2 = dummy_class._process_patient_for_lookup("pat1", [False, False, False])
  assert all(isinstance(x, pd.DataFrame) for x in df_list2)

  # test _process_patient_chunk_for_lookup
  df_list3 = dummy_class._process_patient_chunk_for_lookup(["pat1"], [False, False, False])
  assert all(isinstance(x, pd.DataFrame) for x in df_list3)

  # test create_preset_lookup single-core path (with save_as)
  out_path = tmp_path / "lookup.pkl"
  df_final = dummy_class.create_preset_lookup(None, save_as=str(out_path), cores=None)
  assert out_path.exists()
  assert isinstance(df_final, pd.DataFrame)

  # test empty return (when entry_rows empty)
  with patch.object(dummy_class, "_process_patient_for_lookup", return_value=[]):
    df_empty = dummy_class.create_preset_lookup(["pat1"], cores=None)
    assert df_empty.empty

@patch("physioprep.mimic_iii_ms_tk.module.fetch_with_retry")
def test_full_coverage_multi_core(mock_fetch, dummy_class):
  def side_effect(func, *args, **kwargs):
    if func.__name__ == "get_patients":
      return ["pat1", "pat2"]
    if func.__name__ == "get_records":
      return ["rec1"]
    if func.__name__ == "get_record_segments":
      return (["seg1"], [5])
    if func.__name__ == "get_signals_within":
      return [0.5]
    if func.__name__ == "contains_certain":
      return False
    if func.__name__ == "get_patient_group_id":
      return ("grp", "pid")
    raise ValueError("unexpected call")
  mock_fetch.side_effect = side_effect

  # Patch executor to simulate parallel execution
  future = Future()
  df = pd.DataFrame({"a": [1]})
  future.set_result([df])
  with patch("physioprep.mimic_iii_ms_tk.module.ProcessPoolExecutor") as MockExec:
    MockExec.return_value.__enter__.return_value.submit.return_value = future
    MockExec.return_value.__enter__.return_value.__exit__.return_value = None
    df_final = dummy_class.create_preset_lookup(["pat1", "pat2"], cores=2)
    assert isinstance(df_final, pd.DataFrame)


@patch("physioprep.mimic_iii_ms_tk.module.fetch_with_retry")
def test_create_preset_lookup_exception_branch_only(mock_fetch, capsys):
  # minimal fetch responses to let create_preset_lookup reach multi-core branch
  def side_effect(func, *args, **kwargs):
    name = func.__name__
    if name == "get_patients":
      return ["pat1", "pat2"]
    if name == "get_records":
      return ["rec1"]
    if name == "get_record_segments":
      return (["seg1"], [5])
    if name == "get_signals_within":
      return [0.5]
    if name == "contains_certain":
      return False
    if name == "get_patient_group_id":
      return ("grp", "pid")
    raise ValueError("unexpected call")
  mock_fetch.side_effect = side_effect

  inst = pp.M3WaveFormMasterClass()

  # prepare a Future that is marked done but raises when .result() is called
  bad_future = Future()
  bad_future.set_result(None)  # mark as done so as_completed doesn't block
  def bad_result(timeout=None):
    raise RuntimeError("fake processing error")
  bad_future.result = bad_result

  with patch("physioprep.mimic_iii_ms_tk.module.ProcessPoolExecutor") as MockExec:
    mock_exec_instance = MockExec.return_value.__enter__.return_value
    mock_exec_instance.submit.return_value = bad_future

    # run with cores>1 to reach the ProcessPoolExecutor branch
    df = inst.create_preset_lookup(["pat1", "pat2"], cores=2)

    # capture and check printed exception message
    captured = capsys.readouterr()
    assert "Failed to process chunk" in captured.out
    assert isinstance(df, pd.DataFrame)
    assert df.empty


## -- tests for advanced builtin filter -- ##
def test_get_patient_with_signal_filters_min_samples(monkeypatch):
  module = pp.M3WaveFormMasterClass()
  df = module.preset_metadata
  inp_channels, out_channels = ["II", "PLETH"], ["ABP"]
  df = module.get_patient_with_signal(df, inp_channels = inp_channels, inp_type = "any",
                                      out_channels = out_channels, out_type = "all", min_samples = 20000)
  
  assert df.iloc[0]["segment_len"] >= 20000

## -- tests for get all available signals -- ##
def test_get_available_signals():
  module = pp.M3WaveFormMasterClass()
  available_signals = module.get_available_signals()
  assert set(["II", "ABP", "PLETH"]).issubset(available_signals)

## -- tests for get patient record -- ##
def test_get_patient_record():
  module = pp.M3WaveFormMasterClass()
  rec = module.get_patient_record(group = "p00", pid = "p000020", record_segment = "3544749_0005", 
                                  sampfrom = 10000, sampto = 10125, channels = ['II', 'ABP'])
  
  assert len(rec.p_signal[0]) == 2
  assert len(rec.p_signal[:, 0]) == 125

## -- tests for get data batch -- ##
@pytest.fixture(scope="module")
def module_and_df():
  module = pp.M3WaveFormMasterClass()

  df = module.preset_metadata
  inp_channels, out_channels = ["II", "PLETH"], ["ABP"]
  df = module.get_patient_with_signal(
    df, inp_channels = inp_channels, inp_type = "any",
    out_channels = out_channels, out_type = "all", min_samples = 20000
  )
  return module, df, inp_channels + out_channels

def test_get_data_batch_real(module_and_df):
  module, df, channels = module_and_df
  batch_size, seq_len = 2, 500

  batch, batch_channels, batch_masks, sub_df = module.get_data_batch(
    df, batch_size = batch_size, seq_len = seq_len, 
    channels = channels, sample_res = 32, num_cores = 2, timeout = 3
  )

  assert batch.shape == (batch_size, len(channels), seq_len)
  assert len(batch_channels) == batch_size
  assert len(batch_masks) == batch_size
  assert not np.isnan(batch).any()

  for mask in batch_masks:
    assert mask.dtype == bool
    assert mask.shape[0] == len(channels)


def dummy_get_patient_header(group, pid, segment):
  class Header:
    def __init__(self):
      self.sig_len = 800
  return Header()


def dummy_get_patient_record(group, pid, segment, sampfrom, sampto, sample_res, channels):
  class Record:
    def __init__(self):
      self.p_signal = np.random.randn(750, len(channels))
  return Record()


class DummyValidator:
  def __init__(self, return_value=True):
    self.return_value = return_value
  def apply(self, arr):
    return self.return_value


@pytest.fixture
def dummy_df():
  return pd.DataFrame([
    {
      "patient_group": "A",
      "patient_id": "1",
      "segment": "seg1",
      "signals": ["ch1", "ch2"]
    }
  ])


@pytest.fixture
def dummy_class(monkeypatch):
  obj = pp.M3WaveFormMasterClass()

  obj.get_patient_header = dummy_get_patient_header
  obj.get_patient_record = dummy_get_patient_record
  monkeypatch.setattr(pp, "M3WaveFormValidationModule", lambda: DummyValidator(return_value=True))
  return obj


def test_get_data_batch_single_core(dummy_class, dummy_df):
  """Covers single-core execution."""
  batch, channels, masks, rows = dummy_class.get_data_batch(
    dummy_df,
    batch_size=2,
    seq_len=200,
    channels=["ch1", "ch2", "ch3"],
    sample_res=32,
    num_cores=1,
    timeout=2
  )

  assert isinstance(batch, np.ndarray)
  assert len(channels) == 2
  assert isinstance(rows, pd.DataFrame)
  assert batch.shape[1] == 3


def test_get_data_batch_multi_core(dummy_class, dummy_df):
  """Covers multi-core Pool execution."""
  batch, channels, masks, rows = dummy_class.get_data_batch(
    dummy_df,
    batch_size=2,
    seq_len=100,
    channels=["ch1", "ch2"],
    sample_res=16,
    num_cores=2,
    timeout=2
  )

  assert isinstance(batch, np.ndarray)
  assert batch.ndim == 3
  assert isinstance(rows, pd.DataFrame)
  assert all(isinstance(m, np.ndarray) for m in masks)
