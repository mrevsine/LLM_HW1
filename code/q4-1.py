###=============================================================================
### Imports

from datasets import load_dataset
import torch

###=============================================================================
### Functions

def load_SST2_dataset():
    ds = load_dataset("stanfordnlp/sst2")
    return ds

###=============================================================================
### Main

# Load SST2 dataset
ds = load_SST2_dataset()
print(type(ds))
print(ds.shape)
