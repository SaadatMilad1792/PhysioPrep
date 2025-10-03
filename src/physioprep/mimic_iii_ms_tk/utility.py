########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import time
import requests
import numpy as np
import pandas as pd

########################################################################################################################
## -- supporting utility functions -- ##################################################################################
########################################################################################################################

## -- splits dataframe into smaller independent dataframes -- ##
def get_subject_split(df: pd.DataFrame, split_ratio: list[int] = [1.0],
                      seed: int | None = None) -> list[pd.DataFrame]:
  split_ratio = np.array(split_ratio, dtype=float)
  split_ratio = split_ratio / split_ratio.sum()
  patients = df["patient_id"].unique()
  rng = np.random.default_rng(seed)
  rng.shuffle(patients)

  slices = np.cumsum(len(patients) * split_ratio).astype(int)
  slices = [None] + slices.tolist()
  patient_groups = [patients[slices[i]:slices[i + 1]] for i in range(len(slices) - 1)]
  df_split = [df[df["patient_id"].isin(group)].reset_index(drop = True) for group in patient_groups]
  return df_split

## -- custom waveform selection tool -- ##
def filter_patient_waveforms(df: pd.DataFrame, channels: list[str], mode: str = "all"):

  def filter_row(row_signals: list[str]):
    signals, channels_set = set(row_signals), set(channels)
    if mode == "any":
      return len(signals & channels_set) > 0
    elif mode == "all":
      return channels_set.issubset(signals)
    else:
      raise ValueError(f"Unknown mode: {mode}")
    
  mask = df['signals'].apply(filter_row)
  filtered = df[mask].copy()
  return filtered

## -- safe request retry -- ##
def fetch_with_retry(func, *args, retries: int = 10, delay: int = 5, **kwargs):
  for i in range(retries):
    try:
      return func(*args, **kwargs)
    except (requests.ConnectionError, requests.Timeout) as e:
      print(f"Retry {i + 1}/{retries} failed for {func.__name__} with args {args}... waiting {delay}s")
      time.sleep(delay)
  raise Exception(f"Failed after {retries} retries for {func.__name__} with args {args}")