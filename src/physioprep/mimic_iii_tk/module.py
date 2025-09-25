########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import re
import wfdb
import pooch
import requests
import pandas as pd
from .utility import *
from .validation import *
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
    self.preset_metadata = self.get_preset()

    # self.validation = M3ValidationMasterClass()
    self.args = {
      "dat_cache_dir": pooch.os_cache('wfdb'),
      "physionet_url": "https://physionet.org/files/",
      "physionet_dir": "mimic3wdb-matched/1.0/",
    }

  ## -- get all the available signals -- ##
  def get_available_signals(self, forbidden: list[str] = ['???', '[5125]', '!', '[0]']) -> list[str]:
    unique_signals = self.preset_metadata["signals"].explode().dropna().unique()
    return [s for s in unique_signals if s not in forbidden]
  
  ## -- get preset metadata -- ##
  def get_preset(self) -> pd.DataFrame:
    with resources.open_binary("physioprep.mimic_iii_tk.data", "preset.pkl.gz") as file:
      preset_metadata = pd.read_pickle(file, compression = "gzip")
    return preset_metadata.reset_index(drop = True)

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
  def get_patient_header(self, group: str, pid: str, record: str) -> wfdb.Record:
    pn_dir = f"{self.args['physionet_dir']}{group}/{pid}/"
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
  def contains_certain(self, patient_group_id: str, record_segment: str) -> bool:
    segment_url = f"{self.args['physionet_url']}{self.args['physionet_dir']}{patient_group_id}{record_segment}.hea"
    segment_list = requests.get(segment_url).text.strip().split("\n")
    return not any("uncertain" in s.lower() for s in segment_list)

  ## -- get signals existing within a specific segment -- ##
  def get_signals_within(self, patient_group_id: str, record_segment: str) -> list[str]:
    group, pid = self.get_patient_group_id(patient_group_id)
    header = self.get_patient_header(group, pid, record_segment)
    return list(header.sig_name)
  
  ## -- create the preset lookup table -- ##
  def create_preset_lookup(self, patients: list[str] | None = None, save_as: str | None = None,
                           tqdm_depth: int = 3, cores: int | None = None) -> pd.DataFrame:

    patients, entry_rows = self.get_patients() if patients is None else patients, []
    tqdm_flag = [True if k < tqdm_depth else False for k in range(3)]

    def process_segment(patient, record, segment, segment_len):
      signals = fetch_with_retry(self.get_signals_within, patient, segment)
      certain = fetch_with_retry(self.contains_certain, patient, segment)
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

  ## -- get patients that have the listed signals available -- ##
  def get_patient_with_signal(self, df: pd.DataFrame, inp_channels: list[str] | None = None,
                              inp_type: str = 'any', out_channels: list[str] | None = None,
                              out_type: str = 'any', min_samples: int | None = None) -> pd.DataFrame:
    
    tmp_df = df[df["segment_len"] >= min_samples] if min_samples is not None else df
    tmp_df = filter_patient_waveforms(tmp_df, inp_channels, inp_type) if inp_channels is not None else tmp_df
    tmp_df = filter_patient_waveforms(tmp_df, out_channels, out_type) if out_channels is not None else tmp_df
    return tmp_df
    
  ## -- get patient record as a dataset -- ##
  def get_patient_record(self, group: str, pid: str, record_segment: str, sampfrom: int = 0, sampto: int | None = None, 
                         sample_res: int = 64, channels: list[int] | None = None) -> wfdb.Record:

    df = self.preset_metadata.copy()
    available_channels = df[(df["patient_id"] == pid) & (df["segment"] == record_segment)].iloc[0]
    channels = channels if channels is not None else available_channels["signals"]
    channels = [available_channels["signals"].index(item) for item in channels]
    pn_dir = self.args["physionet_dir"] + group + "/" + pid
    rec = wfdb.rdrecord(record_segment, pn_dir = pn_dir, sampfrom = sampfrom, 
                        sampto = sampto, return_res = sample_res, channels = channels)
    return rec

  ## -- selects a random batch from the data -- ##
  def get_data_batch(self, df: pd.DataFrame, batch_size: int, seq_len: int, 
                     channels: list[str] | None = None, sample_res: int = 64,
                     num_cores: int | None = None, timeout: int = 5) -> tuple[np.ndarray, list, list]:

    validator = M3WaveFormValidationModule()

    def process_row(_):
      for attempt in range(timeout):
        row = df.sample(n = 1).iloc[0]

        header = self.get_patient_header(row["patient_group"], row["patient_id"], row["segment"])
        random_offset = np.random.randint(0, max(0, header.sig_len - seq_len) + 1)
        sampfrom, sampto = random_offset, random_offset + seq_len

        masked_channels = np.array([False if sig in row["signals"] else True for sig in channels]).astype(bool)
        shared_channels = [sig for sig in channels if sig in row["signals"]]

        rec = self.get_patient_record(
          row["patient_group"], row["patient_id"], row["segment"],
          sampfrom=sampfrom, sampto=sampto, sample_res=sample_res, channels=shared_channels
        )

        waveform = rec.p_signal
        masked_waveform = np.zeros((waveform.shape[0], len(masked_channels)))
        masked_waveform[:, ~masked_channels] = waveform[:, :np.sum(~masked_channels)]
        masked_waveform = masked_waveform.transpose(1, 0)

        existing_waveform = masked_waveform[~masked_channels, :]
        if validator.apply(existing_waveform):
          return masked_waveform, shared_channels, masked_channels


      return None

    if num_cores is None or num_cores == 1:
      results = [process_row(None) for _ in range(batch_size)]
    else:
      with ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_row, range(batch_size)))

    final_batch, batch_channels_list, batch_masks_list = [], [], []
    for r in results:
      if r is None:
        zeros_waveform = np.zeros((len(channels), seq_len))
        final_batch.append(zeros_waveform)
        batch_channels_list.append(channels)
        batch_masks_list.append(np.array([False]*len(channels)))
      else:
        final_batch.append(r[0])
        batch_channels_list.append(r[1])
        batch_masks_list.append(r[2])

    return np.stack(final_batch), batch_channels_list, batch_masks_list



  # def get_data_batch(self, df: pd.DataFrame, batch_size: int, seq_len: int, 
  #                  channels: list[str] | None = None, sample_res: int = 64,
  #                  num_cores: int | None = None) -> tuple[np.ndarray, list, list]:

  #   def process_row(row):
  #     header = self.get_patient_header(row["patient_group"], row["patient_id"], row["segment"])
  #     random_offset = np.random.randint(0, max(0, header.sig_len - seq_len) + 1)
  #     sampfrom, sampto = random_offset, random_offset + seq_len
  #     masked_channels = np.array([False if sig in row["signals"] else True for sig in channels]).astype(bool)
  #     shared_channels = [sig for sig in channels if sig in row["signals"]]
  #     rec = self.get_patient_record(
  #       row["patient_group"], row["patient_id"], row["segment"],
  #       sampfrom = sampfrom, sampto = sampto, sample_res = sample_res, channels = shared_channels
  #     )
  #     waveform = rec.p_signal
  #     masked_waveform = np.zeros((waveform.shape[0], len(masked_channels)))
  #     masked_waveform[:, ~masked_channels] = waveform[:, :np.sum(~masked_channels)]
  #     return masked_waveform.transpose(1, 0), shared_channels, masked_channels

  #   rand_rows = df.sample(n = batch_size)

  #   if num_cores is None or num_cores == 1:
  #     results = [process_row(rand_rows.iloc[i]) for i in range(batch_size)]
  #   else:
  #     with ThreadPoolExecutor(max_workers = num_cores) as executor:
  #       results = list(executor.map(process_row, [rand_rows.iloc[i] for i in range(batch_size)]))

  #   batch, batch_channels, batch_masks = zip(*results)
  #   return np.stack(batch), list(batch_channels), list(batch_masks)
