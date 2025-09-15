########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import pandas as pd
import physioprep as pp
# records = module.get_records(patients[0])
# segments, seg_len = module.get_record_segments(patients[0], records[0])
# names = module.get_signals_within(patients[0], segments[0])

########################################################################################################################
## -- tests for physioprep/mimic_iii_tk/module.py -- ###################################################################
########################################################################################################################
def ppm():
  return pp.M3WaveFormMasterClass()

def test_getpatients():
  patients = ppm().get_patients()
  assert "p00/p000020/" in patients

def test_get_patient_group_id():
  group, pid = ppm().get_patient_group_id("p00/p000020/")
  assert group == "p00"
  assert pid == "p000020"

def test_get_records():
  records = ppm().get_records("p00/p000020/")
  assert type(records) == list

def test_get_patient_header():
  records = ppm().get_records("p00/p000020/")
  header = ppm().get_patient_header("p00/p000020/", records[0])
  assert type(header.sig_len) == int

def test_get_record_segments():
  records = ppm().get_records("p00/p000020/")
  segments, seg_len = ppm().get_record_segments("p00/p000020/", records[0])
  assert type(segments[0]) == str
  assert type(seg_len[0]) == str

def test_contains_uncertain():
  records = ppm().get_records("p00/p000020/")
  segments, seg_len = ppm().get_record_segments("p00/p000020/", records[0])
  certain = ppm().contains_uncertain("p00/p000020/", segments[0])
  assert type(certain) == bool

def test_get_signals_within():
  records = ppm().get_records("p00/p000020/")
  segments, seg_len = ppm().get_record_segments("p00/p000020/", records[0])
  signals = ppm().get_signals_within("p00/p000020/", segments[0])
  assert type(signals) == list

def test_create_preset_lookup():
  df = ppm().create_preset_lookup(patients = [], tqdm_depth = 0, 
                                  save_as = None)
  assert len(df) == 0
  
  import os
  df = ppm().create_preset_lookup(patients = ["p00/p000020/"], tqdm_depth = 0, 
                                  save_as = "./create_preset_lookup_tdf.pkl")
  os.remove("./create_preset_lookup_tdf.pkl")
  assert len(df) != 0
  assert type(df) == pd.DataFrame