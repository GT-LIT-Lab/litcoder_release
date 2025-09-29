# LeBel Setup

This guide describes how to prepare LeBel data for LITcoder experiments.

## Requirements
- fMRIPrep preprocessed BOLD in MNI152NLin2009cAsym at 2 mm (res-2)
- Post-processing script: `fmri_processing/to_mni_lebel.py`
- Dataset-level files placed in `data/lebel/neural_data/`:
  - `grids_huge.jbl`
  - `trfiles_huge.jbl`
- Per-subject pickles placed directly in `data/lebel/neural_data/`:
  - `noslice_sub-<SUBJECT_ID>_story_data.pkl`
  - `noslice_sub-<SUBJECT_ID>_story_data_surface.pkl`


## Steps
1) Run fMRIPrep as usual.
2) Run the post-processing script:
```bash
python fmri_processing/to_mni_lebel.py
```
3) Move the produced per-subject pickle files into `data/lebel/neural_data/`.

## Expected Locations
```bash
<repo_root>/data/lebel/neural_data/lebel_data.pkl
<repo_root>/data/lebel/neural_data/noslice_sub-<SUBJECT_ID>_story_data.pkl
<repo_root>/data/lebel/neural_data/noslice_sub-<SUBJECT_ID>_story_data_surface.pkl
```
