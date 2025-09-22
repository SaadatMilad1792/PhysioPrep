########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import wfdb
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import physioprep as pp

########################################################################################################################
## -- tests for physioprep/mimic_iii_tk/module.py -- ###################################################################
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
  with patch("physioprep.mimic_iii_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients = ['p1'], cores = None)
    assert not df.empty
    expected_cols = ['patient_group', 'patient_id', 'record', 'segment', 'certain', 'segment_len', 'signals']
    assert all(col in df.columns for col in expected_cols)

def test_create_preset_lookup_with_save(mock_obj, tmp_path):
  save_file = tmp_path / "lookup.pkl"
  with patch("physioprep.mimic_iii_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients=['p1'], cores = 1, save_as = str(save_file))
    assert save_file.exists()
    expected_cols = ['patient_group', 'patient_id', 'record', 'segment', 'certain', 'segment_len', 'signals']
    assert all(col in df.columns for col in expected_cols)

def test_create_preset_lookup_threaded(mock_obj):
  with patch("physioprep.mimic_iii_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients = ['p1', 'p2'], cores=2)
    assert not df.empty
    expected_cols = ['patient_group', 'patient_id', 'record', 'segment', 'certain', 'segment_len', 'signals']
    assert all(col in df.columns for col in expected_cols)

def test_create_preset_lookup_empty_result(mock_obj):
  mock_obj.get_records = MagicMock(return_value=[])
  with patch("physioprep.mimic_iii_tk.utility.fetch_with_retry", side_effect = lambda f, *a, **k: f(*a, **k)):
    df = mock_obj.create_preset_lookup(patients = ['p1'])
    assert df.empty

def test_create_preset_lookup_threaded_exceptions(capsys):
  obj = pp.M3WaveFormMasterClass()

  patients = ["good_patient", "bad_patient"]

  obj.get_patients = MagicMock(return_value = patients)

  def get_records_side_effect(patient):
    if patient == "bad_patient":
      raise Exception("fail_records")
    return ["record1"]

  obj.get_records = MagicMock(side_effect = get_records_side_effect)
  obj.get_record_segments = MagicMock(return_value = (["seg1"], [100]))
  obj.get_signals_within = MagicMock(return_value = "signal_data")
  obj.contains_certain = MagicMock(return_value = True)
  obj.get_patient_group_id = MagicMock(return_value = ("groupA", "pid123"))

  with patch("tqdm.tqdm", lambda x, **kwargs: x):
    df = obj.create_preset_lookup(patients=None, cores = 2, tqdm_depth = 3)
  assert isinstance(df, pd.DataFrame)
  assert len(df) == 1

  row = df.iloc[0]
  assert row["patient_group"] == "groupA"
  assert row["patient_id"] == "pid123"
  assert row["signals"] == "signal_data"
  assert bool(row["certain"])

  captured = capsys.readouterr()
  assert "Failed to process patient bad_patient: fail_records" in captured.out

  df = obj.create_preset_lookup(patients = None, cores = 2, tqdm_depth = 0)
  assert isinstance(df, pd.DataFrame)
  assert len(df) == 1

  row = df.iloc[0]
  assert row["patient_group"] == "groupA"
  assert row["patient_id"] == "pid123"
  assert row["signals"] == "signal_data"
  assert bool(row["certain"])

  captured = capsys.readouterr()
  assert "Failed to process patient bad_patient: fail_records" in captured.out

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

  # select realistic patient data
  df = module.preset_metadata
  inp_channels, out_channels = ["II", "PLETH"], ["ABP"]
  df = module.get_patient_with_signal(
    df, inp_channels=inp_channels, inp_type="any",
    out_channels=out_channels, out_type="all", min_samples=20000
  )
  return module, df, inp_channels + out_channels

def test_get_data_batch_real(module_and_df):
  module, df, channels = module_and_df
  batch_size, seq_len = 2, 500  # smaller for testing

  batch, batch_channels, batch_masks = module.get_data_batch(
    df, batch_size=batch_size, seq_len=seq_len, channels=channels, sample_res=32, num_cores=2, timeout=3
  )

  # check shapes
  assert batch.shape == (batch_size, len(channels), seq_len)
  assert len(batch_channels) == batch_size
  assert len(batch_masks) == batch_size

  # ensure no NaNs
  assert not np.isnan(batch).any()

  # ensure masks are boolean
  for mask in batch_masks:
    assert mask.dtype == bool
    assert mask.shape[0] == len(channels)

def test_get_data_batch_nan_fallback(module_and_df):
  module, df, channels = module_and_df
  batch_size, seq_len = 2, 300

  # Monkeypatch get_patient_record to inject NaNs for first patient
  original_get_patient_record = module.get_patient_record
  def nan_record(*args, **kwargs):
    rec = original_get_patient_record(*args, **kwargs)
    rec.p_signal[0, 0:10] = np.nan  # insert NaNs into first 10 samples of first channel
    return rec
  module.get_patient_record = nan_record

  batch, batch_channels, batch_masks = module.get_data_batch(
    df, batch_size=batch_size, seq_len=seq_len, channels=channels, timeout=2
  )

  # batch should still have correct shape
  assert batch.shape == (batch_size, len(channels), seq_len)

  # NaNs should be replaced with zeros
  assert not np.isnan(batch).any()
