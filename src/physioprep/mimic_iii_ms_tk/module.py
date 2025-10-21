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
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    with resources.open_binary("physioprep.mimic_iii_ms_tk.data", "preset.pkl.gz") as file:
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
  def _process_segment_for_lookup(self, patient: str, record: str, segment: str, segment_len: int) -> pd.DataFrame:
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

  def _process_record_for_lookup(self, patient: str, record: str, tqdm_flag: list[bool]) -> list[pd.DataFrame]:
    segments, seg_len = fetch_with_retry(self.get_record_segments, patient, record)
    segment_dfs: list[pd.DataFrame] = []
    iterable = zip(segments, seg_len)
    if tqdm_flag[2]:
      iterable = tqdm(list(iterable), desc=f"Segments (record {record})", leave=False)
    for segment, segment_len in iterable:
      segment_dfs.append(self._process_segment_for_lookup(patient, record, segment, segment_len))
    return segment_dfs

  def _process_patient_for_lookup(self, patient: str, tqdm_flag: list[bool]) -> list[pd.DataFrame]:
    records = fetch_with_retry(self.get_records, patient)
    patient_dfs: list[pd.DataFrame] = []
    iterable = records
    if tqdm_flag[1]:
      iterable = tqdm(records, desc=f"Records (patient {patient})", leave=False)
    for record in iterable:
      patient_dfs.extend(self._process_record_for_lookup(patient, record, tqdm_flag))
    return patient_dfs

  def _process_patient_chunk_for_lookup(self, patient_chunk: list[str], tqdm_flag: list[bool]) -> list[pd.DataFrame]:
    results: list[pd.DataFrame] = []
    for patient in patient_chunk:
      results.extend(self._process_patient_for_lookup(patient, tqdm_flag))
    return results

  def create_preset_lookup(self, patients: list[str] | None = None, save_as: str | None = None,
                          tqdm_depth: int = 3, cores: int | None = None) -> pd.DataFrame:

    patients = self.get_patients() if patients is None else patients
    entry_rows: list[pd.DataFrame] = []
    tqdm_flag = [True if k < tqdm_depth else False for k in range(3)]

    def chunk_list(lst: list, n: int) -> list[list]:
      k, m = divmod(len(lst), n)
      return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    if cores is None or cores <= 1:
      iterable = patients
      if tqdm_flag[0]:
        from tqdm.auto import tqdm
        iterable = tqdm(patients, desc="Patients")
      for patient in iterable:
        entry_rows.extend(self._process_patient_for_lookup(patient, tqdm_flag))
    else:
      patient_chunks = chunk_list(patients, cores)
      with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [
          executor.submit(self._process_patient_chunk_for_lookup, chunk, tqdm_flag) for chunk in patient_chunks
        ]
        for f in as_completed(futures):
          try:
            entry_rows.extend(f.result())
          except Exception as e:
            print(f"Failed to process chunk: {e}")

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
  def get_patient_record(self, group: str, pid: str, record_segment: str, sampfrom: int = 0, 
                         sampto: int | None = None, sample_res: int = 64, 
                         channels: list[int] | None = None) -> wfdb.Record:

    df = self.preset_metadata.copy()
    available_channels = df[(df["patient_id"] == pid) & (df["segment"] == record_segment)].iloc[0]
    all_signals = available_channels["signals"]
    channels = channels if channels is not None else all_signals

    seen = set()
    unique_channels = []
    for ch in channels:
      if ch not in seen:
        seen.add(ch)
        unique_channels.append(ch)

    unique_indices = [all_signals.index(ch) for ch in unique_channels]
    pn_dir = self.args["physionet_dir"] + group + "/" + pid
    rec = wfdb.rdrecord(record_segment, pn_dir = pn_dir, sampfrom = sampfrom, 
                        sampto = sampto, return_res = sample_res, channels = unique_indices)

    sig_map = {ch: rec.p_signal[:, i] for i, ch in enumerate(unique_channels)}
    rec.p_signal = np.stack([sig_map[ch] for ch in channels], axis = 1)
    rec.sig_name = channels

    return rec

  ## -- selects a random batch from the data -- ##
  def _process_chunk_for_data_batch(self, rows_chunk, channels, seq_len, sample_res, validator,
                                    timeout, get_patient_header, get_patient_record) -> list:
    chunk_results = []
    for idx, row in rows_chunk:
      for attempt in range(timeout):
        header = get_patient_header(row["patient_group"], row["patient_id"], row["segment"])
        random_offset = np.random.randint(0, max(0, header.sig_len - seq_len) + 1)
        sampfrom, sampto = random_offset, random_offset + seq_len

        masked_channels = np.array(
          [False if sig in row["signals"] else True for sig in channels]
        ).astype(bool)
        shared_channels = [sig for sig in channels if sig in row["signals"]]

        rec = get_patient_record(
          row["patient_group"], row["patient_id"], row["segment"],
          sampfrom=sampfrom, sampto=sampto,
          sample_res=sample_res, channels=shared_channels
        )

        waveform = rec.p_signal
        masked_waveform = np.zeros((waveform.shape[0], len(masked_channels)))
        masked_waveform[:, ~masked_channels] = waveform[:, :np.sum(~masked_channels)]
        masked_waveform = masked_waveform.transpose(1, 0)

        existing_waveform = masked_waveform[~masked_channels, :]
        if validator.apply(existing_waveform):
          chunk_results.append((idx, masked_waveform, shared_channels, masked_channels, row))
          break
      else:
        zeros_waveform = np.zeros((len(channels), seq_len))
        chunk_results.append((idx, zeros_waveform, channels,
                              np.array([False] * len(channels)), row))
    return chunk_results


  def get_data_batch(self, df: pd.DataFrame, batch_size: int, seq_len: int,
                     channels: list[str] | None = None, sample_res: int = 64,
                     num_cores: int | None = None, timeout: int = 5) \
                     -> tuple[np.ndarray, list, list, pd.DataFrame]:

    validator = M3WaveFormValidationModule()

    # Tag rows with their original order index for deterministic merging
    sampled_rows = df.sample(n=batch_size, replace=True).reset_index(drop=True)
    indexed_rows = list(enumerate(sampled_rows.to_dict(orient="records")))

    if num_cores is None or num_cores <= 1:
      results = self._process_chunk_for_data_batch(
        indexed_rows, channels, seq_len, sample_res, validator, timeout,
        self.get_patient_header, self.get_patient_record
      )
    else:
      # Split indexed rows evenly to preserve contiguous segments
      chunks = np.array_split(indexed_rows, num_cores)
      with Pool(processes=num_cores) as pool:
        results_chunks = pool.starmap(
          self._process_chunk_for_data_batch,
          [(chunk.tolist(), channels, seq_len, sample_res, validator, timeout,
            self.get_patient_header, self.get_patient_record) for chunk in chunks]
        )
      results = [item for chunk in results_chunks for item in chunk]

    # Sort by index to ensure correct ordering
    results.sort(key=lambda x: x[0])

    # Unpack aligned results
    final_batch = np.stack([r[1] for r in results])
    batch_channels_list = [r[2] for r in results]
    batch_masks_list = [r[3] for r in results]
    batch_rows_df = pd.DataFrame([r[4] for r in results])

    return final_batch, batch_channels_list, batch_masks_list, batch_rows_df
