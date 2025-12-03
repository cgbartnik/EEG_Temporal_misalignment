# EEG Temporal Misalignment

This repository provides the code to reproduce the analyses presented in:

**Temporal misalignment in scene perception: Divergent representations of locomotive action affordances in human brain responses and DNNs.**  
Bartnik, C. G., Fraats, E. I., & Groen, I. I. A. (2025). *Cognitive Computational Neuroscience Proceedings (CCN-P).*  
[Paper link](https://openreview.net/pdf?id=6FvgJHC4dq)

The project investigates how locomotive action affordances unfold over time during scene perception using high-density EEG. Time-resolved representational similarity analysis (RSA) and spatiotemporal EEG–fMRI fusion are used to evaluate the temporal emergence of affordance-related representations and to compare these with a set of state-of-the-art deep neural network (DNN) features.

Key findings include:
- Affordance representations emerge rapidly (approximately 200 ms after stimulus onset).  
- These representations are distinct from object-related and low-level (GIST) features.  
- EEG–fMRI spatiotemporal fusion reveals a hierarchical contribution of scene-selective regions OPA and PPA.  
- Current DNNs capture these dynamics only partially, replicating divergences previously observed in fMRI.

If you encounter issues with the code, please open an issue or submit a pull request.

Raw EEG data used in this study is publicly available on OSF:  
[https://osf.io/v3rcq/overview](https://osf.io/v3rcq/overview)

(Also download and extract `Additional_experimental_files.zip` from OSF; it contains model RDMs used in the analyses.)


## Repository layout
- `00_preprocessing/`: script for raw EEG cleaning, filtering, and epoching.
- `01_Affordance_vs_Gist/`: experiments comparing affordance and gist representations.
- `02_Spatiotemporalfusion/`: Spatiotemporal fusion of fMRI and EEG data
- `03_DNN/`: DNN feature alignment with EEG data
- `04_Supplement_analysis/`: follow-up analyses

## Getting started
1. Create the Conda environment: `conda env create -f environment.yml`
2. Activate it: `conda activate eeg-temporal-misalignment`
3. Add your data and notebooks/scripts into the stage folders above.
