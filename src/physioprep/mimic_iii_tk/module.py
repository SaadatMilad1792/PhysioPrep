########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import re
import wfdb
import pooch
import requests
import pandas as pd
from .utility import *
from tqdm.auto import tqdm
from importlib import resources
from concurrent.futures import ThreadPoolExecutor, as_completed

########################################################################################################################
## -- mimic iii toolkit master class module -- #########################################################################
########################################################################################################################

## -- mimic iii toolkit master class for preprocessing -- ##
class M3WaveFormMasterClass():
  def __init__(self) -> None:
    super(M3WaveFormMasterClass, self).__init__()
    # self.validation = M3ValidationMasterClass()

    self.args = {
      "dat_cache_dir": pooch.os_cache('wfdb'),
      "physionet_url": "https://physionet.org/files/",
      "physionet_dir": "mimic3wdb-matched/1.0/",
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
                          tqdm_depth: int = 3, cores: int | None = None) -> pd.DataFrame:

    patients, entry_rows = self.get_patients() if patients is None else patients, []
    tqdm_flag = [True if k < tqdm_depth else False for k in range(3)]

    def process_segment(patient, record, segment, segment_len):
      signals = fetch_with_retry(self.get_signals_within, patient, segment)
      certain = fetch_with_retry(self.contains_uncertain, patient, segment)
      group, pid = fetch_with_retry(self.get_patient_group_id, patient)
      return pd.DataFrame({
        "patient_group": [str(group)],
        "patient_id": [str(pid)],
        "record": [str(record)],
        "segment": [str(segment)],
        "certain": [bool(certain)],
        "segment_len": [int(segment_len)],
        "signals": [signals],
      })

    def process_record(patient, record):
      segments, seg_len = fetch_with_retry(self.get_record_segments, patient, record)
      segment_dfs = []
      iterable = zip(segments, seg_len)
      if tqdm_flag[2]:
        iterable = tqdm(list(iterable), desc = f"Segments (record {record})", leave = False)
      for segment, segment_len in iterable:
        segment_dfs.append(process_segment(patient, record, segment, segment_len))
      return segment_dfs

    def process_patient(patient):
      records = fetch_with_retry(self.get_records, patient)
      patient_dfs = []
      iterable = records
      if tqdm_flag[1]:
        iterable = tqdm(records, desc = f"Records (patient {patient})", leave = False)
      for record in iterable:
        patient_dfs.extend(process_record(patient, record))
      return patient_dfs

    if cores is None or cores <= 1:
      iterable = patients
      if tqdm_flag[0]:
        iterable = tqdm(patients, desc = "Patients")
      for patient in iterable:
        entry_rows.extend(process_patient(patient))
    else:
      with ThreadPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(process_patient, patient): patient for patient in patients}
        if tqdm_flag[0]:
          for f in tqdm(as_completed(futures), total=len(futures), desc = "Patients"):
            try:
              entry_rows.extend(f.result())
            except Exception as e:
              print(f"Failed to process patient {futures[f]}: {e}")
        else:
          for f in as_completed(futures):
            try:
              entry_rows.extend(f.result())
            except Exception as e:
              print(f"Failed to process patient {futures[f]}: {e}")

    if entry_rows:
      df = pd.concat(entry_rows).reset_index(drop = True)
      if save_as:
        df.to_pickle(save_as)
      return df
    return pd.DataFrame()




  # ## -- get all the available signals -- ##
  # def get_available_signals(self) -> list[str]:
  #   forbidden = ['???']
  #   unique_signals = self.args_preset["patients_list"]["patient_signals"].explode().dropna().unique()
  #   return [s for s in unique_signals if s not in forbidden]

  # ## -- get patients that have the listed signals available -- ##
  # def get_patient_with_signal(self, patients: list[str] | None = None, 
  #                             signal_filter: list[str] | None = None) -> pd.DataFrame:
    
  #   df = self.args_preset["patients_list"].copy()
  #   patients = patients if patients is not None else list(df["patient_id"])
  #   df = df[df["patient_id"].isin(patients)]
  #   if signal_filter is not None:
  #     df = df[df["patient_signals"].apply(lambda sig: set(signal_filter).issubset(sig))]
  #   return df

  # ## -- get patient record as a dataset -- ##
  # def get_patient_record(self, group: str, pid: str, record: str, sampfrom: int = 0, 
  #                        sampto: int | None = None, channels: list[int] | None = None) -> wfdb.Record:

  #   df = self.args_preset["patients_list"].copy()
  #   available_channels = df[(df["patient_id"] == pid) & (df["patient_record"] == record)].iloc[0]
  #   channels = channels if channels is not None else available_channels["patient_signals"]
  #   channels = [available_channels["patient_signals"].index(item) for item in channels]
  #   pn_dir = self.args_preset["physionet_dir"] + group + "/" + pid
  #   rec = wfdb.rdrecord(record, pn_dir = pn_dir, sampfrom = sampfrom, sampto = sampto, channels = channels)
  #   return rec

  # ## -- selects a random batch from the data -- ##
  # def get_data_batch(self, df: pd.DataFrame, batch_size: int, signal_len: int, 
  #                    channels: list[str] | None = None, timeout: int = 100) -> np.array:

  #   batch, timeout_counter = [], 0
  #   timeout = max(batch_size, timeout)
  #   while len(batch) < batch_size and timeout_counter <= timeout:
  #     timeout_counter += 1
  #     # timeout_counter += 1 if len(batch) > 0 else 0
  #     rand_row = df.sample(n = 1).iloc[0]
  #     group, pid, record = rand_row["patient_group"], rand_row["patient_id"], rand_row["patient_record"]

  #     try:
  #       header = self.get_patient_header(group, pid, record)
  #       random_offset = np.random.randint(0, max(0, header.sig_len - signal_len) + 1)
  #       sampfrom, sampto = random_offset, random_offset + signal_len
  #       rec = self.get_patient_record(group, pid, record, sampfrom = sampfrom, sampto = sampto, channels = channels)
  #       waveform, val_flag = rec.p_signal, self.validation.apply(rec.p_signal, signal_len)
  #       batch.append(waveform.transpose(1, 0)) if val_flag else None
  #       # print(rec.p_signal.shape, header.fs, rec.sig_name)
  #     except:
  #       continue
    
  #   return np.stack(batch) if len(batch) > 0 else False