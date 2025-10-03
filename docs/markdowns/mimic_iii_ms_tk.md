
## MIMIC-III Toolkit
This section is specifically dedicated to providing a comprehensive and detailed documentation of the MIMIC-III Toolkit, which serves as a set of utilities and methods designed for working with the MIMIC-III Waveform Database Matched Subset. In addition to describing the functionalities and features of the toolkit, this section thoroughly explains the underlying methodologies, the rationale behind each implemented function, and the recommended usage patterns. The goal is to equip researchers, data scientists, and practitioners with clear guidance, practical examples, and contextual information that facilitates the effective and efficient use of the toolkit for tasks such as physiological time-series analysis, preprocessing, and integration with generative modeling workflows. By covering both high-level conceptual overviews and low-level implementation details, this section aims to provide a complete resource for understanding, navigating, and applying the MIMIC-III Toolkit in a variety of research and analytical scenarios.

<!-- get available signals -->
### METHOD &#x279C; get_available_signals
```python
get_available_signals(forbidden: list[str] = ['???', '[5125]', '!', '[0]']) -> list[str]
```
**Description:** gets all available signals in the [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) database across all patients.  
**Parameters:**  
- forbidden: removes unknown or vague signals, the default is `['???', '[5125]', '!', '[0]']`.  

**Returns:** list of all available signals in the entire [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/).  

<!-- get preset -->
### METHOD &#x279C; get_preset
```python
get_preset() -> pd.DataFrame
```
**Description:** gets the preset dataframe with all the patients, records, segments, and their available signals.  
**Returns:** a data frame of all patients, their records, segments, and available signal modalities in each.  

<!-- get patients -->
### METHOD &#x279C; get_patients
```python
get_patients() -> list[str]
```
**Description:** gets a list of all available patients directly from physionet RECORDS file.  
**Returns:** a list of strings with the following format `['p00/p000020/', 'p00/p000030/', ...]`.  

<!-- get patient group id -->
### METHOD &#x279C; get_patient_group_id
```python
get_patient_group_id(patient_group_id: str) -> tuple[str, str]
```
**Description:** extracts the group and id of a patient from a string.  
**Parameters:**  
- patient_group_id: string of patient group_id (e.g. `p00/p000020/`).  

**Returns:** a tuple of strings where the first is group and second is id (e.g. `p00`, and `p000020`).  

<!-- get records -->
### METHOD &#x279C; get_records
```python
get_records(patient_group_id: str) -> list[str]
```
**Description:** gets all available records for a specific patient.  
**Parameters:**  
- patient_group_id: string of patient group_id (e.g. `p00/p000020/`).  

**Returns:** a list of strings, where each entry is a recording of the patient.  

<!-- get patient header -->
### METHOD &#x279C; get_patient_header
```python
get_patient_header(group: str, pid: str, record: str) -> wfdb.Record
```
**Description:** given a patients information, fetches the corresponding header files.  
**Parameters:**  
- group: patients group indicator (e.g. `p00`).  
- pid: patients id indicator (e.g. `p000020`).  
- record: patients record name (e.g. `p000020-2183-04-28-17-47`).  

**Returns:** returns the corresponding `wfdb.Record` for a given patient (just the header).  

<!-- get record segments -->
### METHOD &#x279C; get_record_segments
```python
get_record_segments(patient_group_id: str, record: str) -> tuple[list[str], list[str]]
```
**Description:** extracts all available segments for a given patient record.  
**Parameters:**  
- patient_group_id: patients group_id indicator (e.g. `p00/p000020/`).  
- record: patients record name (e.g. `p000020-2183-04-28-17-47`).  

**Returns:** returns a tuple where the first entry is list of available segments, 
and the second is a list of sample sizes per segment.  

<!-- contains certain -->
### METHOD &#x279C; contains_certain
```python
contains_certain(patient_group_id: str, record_segment: str) -> bool
```
**Description:** checks to see if the segment is aligned with certainty.  
**Parameters:**  
- patient_group_id: patients group_id indicator (e.g. `p00/p000020/`).  
- record_segment: patients record segment name (e.g. `3544749_0001`).  

**Returns:** returns a boolean that is `True` for certain ones and `False` for certain ones.  

<!-- contains certain -->
### METHOD &#x279C; get_signals_within
```python
get_signals_within(patient_group_id: str, record_segment: str) -> list[str]
```
**Description:** gets a list of signals within a segment.  
**Parameters:**  
- patient_group_id: patients group_id indicator (e.g. `p00/p000020/`).  
- record_segment: patients record segment name (e.g. `3544749_0001`).  

**Returns:** returns a list of signals within a segment as a list of strings.  

<!-- create preset lookup -->
### METHOD &#x279C; create_preset_lookup
```python
create_preset_lookup(patients: list[str] | None = None, save_as: str | None = None,
                     tqdm_depth: int = 3, cores: int | None = None) -> pd.DataFrame
```
**Description:** creates a dataframe that includes all patients, records, segments, and available signals in each.  
**Parameters:**  
- patients: a list of patient_group_ids, or `None`, if set to `None`, performs on all patients.  
- save_as: address to the desired save directory, if set to `None` won't save the data frame.  
- tqdm_depth: between 0 to 3. 0 disables the tqdm, 1 is patient level, 2 is record level, and 3 is segment level.
- cores: allows for multicore processing, if set to `None` falls back to single core.

**Returns:** returns a dataframe that includes all patients, records, segments, and available signals in each.  
**Note:** not recommended to run this function, the dataframe can be accessed as `preset_metadata` property of the 
`M3WaveFormMasterClass` module, takes a very long time to run.

<!-- get_patient with signal -->
### METHOD &#x279C; get_patient_with_signal
```python
get_patient_with_signal(df: pd.DataFrame, inp_channels: list[str] | None = None,
                        inp_type: str = 'any', out_channels: list[str] | None = None,
                        out_type: str = 'any', min_samples: int | None = None) -> pd.DataFrame
```
**Description:** takes a dataframe, filters the rows with desired signal modalities in inputs and output.  
**Parameters:**  
- df: input data frame, if set to `None`, uses the `preset_metadata` for all patients.  
- inp_channels: desired signals as input.  
- inp_type: could be either `any` or `all`, `any` will include any row with at least one compatible entry, while
`all` will include the rows that include all the desired inp_channels.  
- out_channels: desired signals as output.  
- out_type: could be either `any` or `all`, `any` will include any row with at least one compatible entry, while
`all` will include the rows that include all the desired out_channels.  
- min_samples: includes only the rows with at least `min_samples` samples in their segments.

**Returns:** returns a dataframe that includes all patients, records, segments, and available signals in each.  
**Note:** not recommended to run this function, the dataframe can be accessed as `preset_metadata` property of the 
`M3WaveFormMasterClass` module, takes a very long time to run.

<!-- get patient record -->
### METHOD &#x279C; get_patient_record
```python
get_patient_record(group: str, pid: str, record_segment: str, sampfrom: int = 0, sampto: int | None = None, 
                   sample_res: int = 64, channels: list[int] | None = None) -> wfdb.Record
```
**Description:** given a patients information, fetches the corresponding record segment data files.  
**Parameters:**
- group: patients group indicator (e.g. `p00`).  
- pid: patients id indicator (e.g. `p000020`).  
- record: patients record name (e.g. `3544749_0001`).  
- sampfrom: starting point of the loaded files, initially set to `0`.  
- sampto: ending point of the loaded files, if set to `None`, will load the entire file.  
- sample_res: resolution of the loaded data, initially set to 64 for double.  
- channels: specific channels to be included in the record file (e.g. `['ABP', 'PLETH']`).  

**Returns:** returns the corresponding `wfdb.Record` for a given patient (just the data file `.dat`).  

<!-- get data batch -->
### METHOD &#x279C; get_data_batch
```python
get_data_batch(df: pd.DataFrame, batch_size: int, seq_len: int, channels: list[str] | None = None, 
               sample_res: int = 64, num_cores: int | None = None, timeout: int = 5) -> tuple[np.ndarray, list, list]
```
**Description:** generates a batch of data with a certain length from a given dataframe.  
**Parameters:**  
- df: the initial dataframe.  
- batch_size: training portion of the dataframe, default value is `0.8`.  
- seq_len: length of the chosen segment for each batch element.  
- channels: signals to be included in the batch (e.g. `['ABP', 'II']`).  
  - **Note:** Only use `None` if you have already filtered the signals during splitting.  
- sample_res: resolution of samples passed to `get_patient_record`, initially set to 64.
- num_cores: allows for multicore processing, if set to `None` falls back to single core.
- timeout: maximum number of attempts before returning all zeros as fallback value.

**Returns:** returns a np.array of shape `(batch_size, len(channels), seq_len)`, and two additional lists, one for masked entries, and one for available signals per entry in batch.

<!-- get subject split -->
### Methods &#x279C; get_subject_split
```python
get_subject_split(df: pd.DataFrame, split_ratio: list[int] = [1.0],
                  seed: int | None = None) -> list[pd.DataFrame]
```
**Description:** splits a dataframe, to patient-based isolated training, testing, and validation sets.

**Parameters:**
- df: the initial dataframe.
- split_ratio: a list of dynamic size for all slices (e.g. `[0.8, 0.1, 0.1]`).
- seed: random_state for reproducibility.

**Returns:** returns a tuple of dataframes of patient info for train, test, and validation, etc.  

## Navigation Panel
<!-- - [Next (TBD)](/docs/markdowns/getting_started.md) -->
- [Return to repository](/)
- [Back (Getting Started)](/docs/markdowns/getting_started.md)