# EEG Temporal Misalignment

Analysis workspace for exploring temporal misalignment effects in EEG signals. The project is organized into stage-specific folders so data preprocessing, model training, and supplementary analyses stay separated.

## Repository layout
- `00_preprocessing/`: scripts and notebooks for raw EEG cleaning, filtering, and epoching.
- `01_Affordance_vs_Gist/`: experiments comparing affordance and gist representations.
- `02_Spatiotemporalfusion/`: feature fusion and temporal alignment experiments.
- `03_DNN/`: deep learning models and training runs.
- `04_Supplement_analysis/`: follow-up analyses, figures, or tables.

## Getting started
1. Create the Conda environment: `conda env create -f environment.yml`
2. Activate it: `conda activate eeg-temporal-misalignment`
3. Add your data and notebooks/scripts into the stage folders above.

## Notes
- Data is not included; place raw and processed EEG files under the appropriate stage directory.
- Add any additional dependencies you use to `environment.yml` and re-export with `conda env export --from-history`.
- Initialize Git remotes as needed: `git remote add origin <url>` then `git push -u origin main`.
