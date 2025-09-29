# fMRI Processing for the LeBel dataset

This folder contains the script we used to process the LeBel dataset. We did this to ensure alignment with the paper: [A natural language fMRI dataset for voxelwise encoding models](https://www.nature.com/articles/s41597-023-02437-z)

## Steps

1. Run fMRIPrep as usual.
2. Run the post-processing script:
```bash
python fmri_processing/to_mni_lebel.py
```
In the script, you have to change the `base_path` and the `subject` variables.