![Python](https://img.shields.io/badge/python-3.10-2ca02c?style=flat&labelColor=2f2f4f)
[![License: MIT](https://img.shields.io/badge/license-MIT-1f77b4?style=flat&labelColor=2f2f4f)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/physioprep?style=flat&labelColor=2f2f4f&color=9467bd&logo=pypi)](https://pypi.org/project/physioprep/)
[![TestPyPI](https://img.shields.io/badge/dynamic/json?url=https://test.pypi.org/pypi/physioprep/json&query=info.version&label=TestPyPI&style=flat&labelColor=2f2f4f&color=9467bd&logo=pypi&prefix=v)](https://test.pypi.org/project/physioprep/)
[![Codecov](https://codecov.io/gh/SaadatMilad1792/physioprep/branch/master/graph/badge.svg)](https://codecov.io/gh/SaadatMilad1792/physioprep)

![physioprep logo](docs/images/physioprep_logo.png)

## Introduction to Physioprep
Physioprep is a toolkit designed to support researchers working with physiological time-series data and generative models. Although still under active development, its long-term goal is to provide a flexible framework capable of handling a wide range of physiological signals. The package was originally conceived as a utility library for training predictive generative models, particularly approaches inspired by Causal Language Modeling (CLM) but applied to physiological waveforms. However, its applications are not limited to that domain, and it can be used for many other research purposes as well. In the longer term, Physioprep is intended to include dedicated toolkits for data cleaning, artifact removal, and quality assessment, making it a comprehensive resource for physiological machine learning research.

## Table of Content
|    | Topic                                                  | Description                                                  | Google Colab                                                                                                                                                                                   |
|----|--------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1 | [Getting Started](./docs/markdowns/getting_started.md) | User guide to setting up and getting started with physioprep | N/A                                                                                                                                                                                            |
|  2 | [MIMIC III Toolkit](./docs/markdowns/mimic_iii_tk.md)  | User guide to MIMIC III Waveform Dataset Matched Subset      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaadatMilad1792/physioprep/blob/master/docs/examples/mimic_iii_tk.ipynb) |

## Acknowledgement
At the time of the initial release, this package is primarily focused on the [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/), one of the largest openly accessible physiological datasets. The goal is to enable its use in autoregressive predictive generative modeling. Development of this project was greatly inspired by the excellent [WFDB](https://wfdb.readthedocs.io/en/latest/index.html) library (Waveform Database). We are grateful for their work, which made it possible to build specialized modules like Physioprep in an efficient and task-oriented way.

## Navigation Panel
- [Next (Getting Started)](/docs/markdowns/getting_started.md)
<!-- - [Return to repository (Disabled)](/) -->
<!-- - [Back (Disabled)](/) -->