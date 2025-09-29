# Little Prince (LPP) Setup

This guide describes how to prepare Little Prince data for LITcoder experiments.

## Requirements
- fMRIPrep preprocessed BOLD for all runs in MNI152NLin2009cAsym at 2 mm (res-2)
- LPP data should be stored in `data/little_prince/neural_data/`

## Expected Layout (example `sub-EN058`)
```bash
<repo_root>/data/little_prince/neural_data/sub-EN058/lppEN_word_information.csv
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-03_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-04_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-05_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-06_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-07_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-08_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
<repo_root>/data/little_prince/neural_data/sub-EN058/sub-EN058_task-lppEN_run-09_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz
```

## Verify
- `lppEN_word_information.csv` present for each subject
- All expected BOLD runs present
