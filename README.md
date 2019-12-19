# Local epigenomic data are more informative than local genome sequence data in predicting enhancer-promoter interactions
This repository shares the CNN implementation codes: Basic CNN, ResNet CNN for both sequence and genomics and attention CNN for sequence data. Both the analysis codes and some data processing codes are provided.

## Requirements
For training scripts:

Python 3.5.4 :: Anaconda, Inc.

Keras 2.0.9 with Tensorflow 1.8.0

For data preprocessing:

Python 2.7.12

## Data
The sequence and epigenomic data we used can be downloaded from https://github.com/menglix/EPI/releases/tag/Data.
## Codes
We should first run:

    python otherFiles.py
    
Then \*_EPI.py/\*_seq.py/Combine_seq_epigenomics.py.
### Match sequence with the epigenomic data
Run: 

    python k562.py 

Then, the match correction requires interactive run of post_match_K562.py to get the one-to-one matched dictionary in Python
### Analysis
1. Sequence training
 
Basic CNN (3000/2000 bp):

    python BasicCNN_seq.py
    
Attention CNN (regional):

    python AttentionCNN_seq.py
    
ResNet CNN (regional):

    python ResNetCNN_seq.py

2. Epigenomic training

Basic CNN:

    python BasicCNN_EPI.py

ResNet CNN:

    python ResNetCNN_EPI.py

3. Combined model (use the matched dictionary between the sequence and epigenomic data)

Combined model (after running sequence and epigenomic model):

    python Combine_seq_epigenomics.py

All codes are self-explanatory with detailed comments. If you have any questions, please contact xiaox345@umn.edu.

