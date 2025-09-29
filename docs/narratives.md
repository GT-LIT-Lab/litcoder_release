# Narratives Setup

This guide describes how to prepare Narratives data for LITcoder experiments.

## Requirements
- fMRIPrep preprocessed BOLD in MNI152NLin2009cAsym at 2 mm (res-2)
- Dataset-level data: `narratives_data.pkl`
- Per-subject BOLD files placed inside every subject folder


## Expected Layout
```bash
# dataset-level
<repo_root>/data/narratives/neural_data/narratives_data.pkl

# per-subject (example)
<repo_root>/data/narratives/neural_data/sub-256/sub-256_task-21styear_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```
