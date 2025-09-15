########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import re
import wfdb
import pooch
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from importlib import resources

########################################################################################################################
## -- mimic iii toolkit master class module -- #########################################################################
########################################################################################################################
class M3WaveFormMasterClass():
  def __init__(self) -> None:
    super(M3WaveFormMasterClass, self).__init__()
    # with resources.open_binary("physioprep.data", "patient_signals.pkl") as file:
    #   patients_list_csv = pd.read_pickle(file)

    # self.validation = M3ValidationMasterClass()
    self.args = {
      "dat_cache_dir": pooch.os_cache('wfdb'),
      "physionet_url": "https://physionet.org/files/",
      "physionet_dir": "mimic3wdb-matched/1.0/",
      # "patients_list": patients_list_csv,
    }

  ## -- get the list of patients from preset .pkl or from physionet -- ##
  def get_patients(self) -> list[str]:
    patients_url = F"{self.args['physionet_url']}{self.args['physionet_dir']}RECORDS"
    patients_list = requests.get(patients_url).text.strip().split("\n")
    return list(patients_list)

  ## -- get the group and id for a single patient entry of form ("pXX/pXXXXXX/") -- ##
  def get_patient_group_id(self, patient_group_id: str) -> tuple[str, str]:
    group, pid = re.match("([^/]+)/([^/]+)/", patient_group_id).groups()
    return group, pid
  
  ## -- get records associated with a patient -- ##
  def get_records(self, patient_group_id: str) -> list[str]:
    records_url = f"{self.args['physionet_url']}{self.args['physionet_dir']}{patient_group_id}RECORDS"
    records_list = requests.get(records_url).text.strip().split("\n")
    pattern = r'^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$'
    records = [r for r in records_list if re.match(pattern, r)]
    return records
  
  ## -- get patient record as a header -- ##
  def get_patient_header(self, patient_group_id: str, record: str) -> wfdb.Record:
    pn_dir = f"{self.args['physionet_dir']}{patient_group_id}"
    header = wfdb.rdheader(record, pn_dir = pn_dir)
    return header
  
  ## -- get segments within each recording -- ##
  def get_record_segments(self, patient_group_id: str, record: str) -> tuple[list[str], list[str]]:
    record_url = f"{self.args['physionet_url']}{self.args['physionet_dir']}{patient_group_id}{record}.hea"
    record_list = requests.get(record_url).text.strip().split("\n")
    pattern = r'^(\d{7}_\d{4}) (.+)$'
    
    record_ids, record_inf = [], []
    for r in record_list:
      r = r.strip()
      match = re.match(pattern, r)
      if match:
        record_ids.append(match.group(1))
        record_inf.append(match.group(2))

    return record_ids, record_inf
  
  ## -- checks if the alignment is uncertain -- ##
  def contains_uncertain(self, patient_group_id: str, record_segment: str) -> bool:
    segment_url = f"{self.args['physionet_url']}{self.args['physionet_dir']}{patient_group_id}{record_segment}.hea"
    segment_list = requests.get(segment_url).text.strip().split("\n")
    return not any("uncertain" in s.lower() for s in segment_list)

  ## -- get signals existing within a specific segment -- ##
  def get_signals_within(self, patient_group_id: str, record_segment: str) -> list:
    header = self.get_patient_header(patient_group_id, record_segment)
    return list(header.sig_name)
  
  ## -- create the preset lookup table -- ##
  def create_preset_lookup(self, patients: list[str] | None = None, save_as: str | None = None, 
                           tqdm_depth: int = 3) -> pd.DataFrame:
    entry_row, tqdm_depth = [], [False if i < tqdm_depth else True for i in range(3)]
    patients = self.get_patients() if patients is None else patients

    for patient in tqdm(patients, desc = "Patients", position = 0, disable = tqdm_depth[0]):
      group, pid = self.get_patient_group_id(patient)
      records = self.get_records(patient)

      for record in tqdm(records, desc = f"Records (patient {pid})", position = 1, 
                         leave = False, disable = tqdm_depth[1]):
        segments, seg_len = self.get_record_segments(patient, record)

        for segment, segment_len in tqdm(list(zip(segments, seg_len)), desc = f"Segments (record {record})", 
                                        position = 2, leave = False, disable = tqdm_depth[2]):
          signals = self.get_signals_within(patient, segment)
          certain = self.contains_uncertain(patient, segment)
          entry_row.append(pd.DataFrame({
            "patient_group": str(group),
            "patient_id": str(pid),
            "record": str(record), 
            "segment": str(segment),
            "certain": bool(certain),
            "segment_len": int(segment_len),
            "signals": [signals],
          }))

    if entry_row:
      df = pd.concat(entry_row).reset_index(drop=True)
      if save_as is not None:
        df.to_pickle(save_as)
      return df
    else:
      return pd.DataFrame()