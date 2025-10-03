## Getting Started

To ensure a clean and reproducible setup, it is strongly recommended to create a dedicated Python environment before installing **Physioprep**. Using an isolated environment helps prevent conflicts with existing packages and makes your workflow more reliable. You can create a new environment using **Conda** as follows:

```bash
# Create a new Conda environment named <env-name> with Python 3.10
conda create --name <env-name> python=3.10

# Activate the newly created environment
conda activate <env-name>
```

Once your environment is active, **PhysioPrep** can be installed easily via `pip` (recommended stable version):

```bash
pip install physioprep==<version>
```

> **Note:** You can also install the latest development version directly from GitHub (bleeding edge).  
> This version may include **experimental or incomplete features**.  
> Use it **only if you understand the risks**:

```bash
pip install git+https://github.com/SaadatMilad1792/PhysioPrep.git
```

After the installation completes, your environment is ready to use **PhysioPrep**. You can start by importing the package in a Python session to verify that it is installed correctly:

```python
import physioprep
print(physioprep.__version__)
```

## Navigation Panel
- [Next (MIMIC III Toolkit)](/docs/markdowns/mimic_iii_ms_tk.md)
- [Return to repository](/)
- [Back (Introduction)](/README.md)
