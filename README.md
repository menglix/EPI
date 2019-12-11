# Local epigenomic data are more informative than local genome sequence data in predicting enhancer-promoter interactions
This repository shares the CNN implementation codes: Basic CNN, ResNet CNN for both sequence and genomics and attention CNN for sequence data. Both the analysis codes and some data processing codes are provided.

## Requirement
For training scripts:

Python 3.5.4 :: Anaconda, Inc.

Keras 2.0.9

For data preprocessing:

Python 2.7.12

## Data
The sequence and epigenomic data we used can be downloaded from https://github.com/menglix/EPI/releases/tag/Data.
## Codes
We should first run otherFiles.py, and then \*_EPI.py/\*_seq.py/Combine_seq_epigenomics.py.
### Match sequence with the epigenomic data
1. k562.py 
2. post_match_K562.py
3. get the one-to-one matched dictionary in Python
### Analysis:
1. Sequence training: BasicCNN_seq.py | AttentionCNN_seq.py | ResNetCNN_seq.py
2. Epigenomic training: BasicCNN_EPI.py | ResNetCNN_EPI.py
3. Combined model: Combine_seq_epigenomics.py (use the matched dictionary between the sequence and epigenomic data)

