import nibabel as nib
import numpy as np
import os
import pickle
from glob import glob
import functools
from dataclasses import dataclass
from nilearn import datasets, surface


@dataclass
class SurfaceData:
    """Data class to hold surface data information."""

    left_hemisphere: np.ndarray
    right_hemisphere: np.ndarray
    combined: np.ndarray


class SurfaceProjector:
    """Handles projection of volumetric data to surface."""

    def __init__(self):
        self.fsaverage = datasets.fetch_surf_fsaverage()
        self.mesh_left = surface.load_surf_mesh(self.fsaverage["pial_left"])
        self.mesh_right = surface.load_surf_mesh(self.fsaverage["pial_right"])

    def project_volume_to_surface(
        self, volume_data: np.ndarray, affine: np.ndarray
    ) -> SurfaceData:
        """Project volumetric data to surface for both hemispheres.

        Args:
            volume_data: 4D numpy array of shape (x, y, z, time)
            affine: Affine transformation matrix

        Returns:
            SurfaceData object containing left, right and combined surface data
        """
        n_timepoints = volume_data.shape[3]
        n_vertices_left = self.mesh_left[0].shape[0]
        n_vertices_right = self.mesh_right[0].shape[0]

        surface_data_left = np.zeros((n_timepoints, n_vertices_left))
        surface_data_right = np.zeros((n_timepoints, n_vertices_right))

        for t in range(n_timepoints):
            vol_t = volume_data[:, :, :, t]
            img_t = nib.Nifti1Image(vol_t, affine)

            data_left = surface.vol_to_surf(img_t, self.mesh_left)
            data_right = surface.vol_to_surf(img_t, self.mesh_right)

            surface_data_left[t, :] = data_left
            surface_data_right[t, :] = data_right

        combined = np.column_stack((surface_data_left, surface_data_right))
        return SurfaceData(surface_data_left, surface_data_right, combined)


# Base paths
base_path = "/storage/coda1/p-aivanova7/0/shared/ds003020_fmriprep_noslice2"
subject = "sub-UTS01"

# Story names
stories = [
    "adollshouse",
    "adventuresinsayingyes",
    "alternateithicatom",
    "avatar",
    "buck",
    "exorcism",
    "eyespy",
    "fromboyhoodtofatherhood",
    "hangtime",
    "haveyoumethimyet",
    "howtodraw",
    "inamoment",
    "itsabox",
    "legacy",
    "naked",
    "odetostepfather",
    "sloth",
    "souls",
    "stagefright",
    "swimmingwithastronauts",
    "thatthingonmyarm",
    "theclosetthatateeverything",
    "tildeath",
    "undertheinfluence",
    "wheretheressmoke",
]


def find_story_files(base_path, subject, story):
    """Find BOLD and brain mask files for a given story"""
    # Search for BOLD files across all sessions (including run numbers for test story)
    bold_pattern = f"{base_path}/{subject}/ses-*/func/{subject}_ses-*_task-{story}_*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    bold_files = glob(bold_pattern)
    print("these are bold files1")
    print(bold_files)

    # Also try pattern without run numbers for regular stories
    if not bold_files:
        bold_pattern_no_run = f"{base_path}/{subject}/ses-*/func/{subject}_ses-*_task-{story}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        bold_files = glob(bold_pattern_no_run)
        print(f"  Tried no-run pattern: {bold_pattern_no_run}")

    # Debug: show exactly what we're looking for
    print(f"  Story: {story}")
    print(f"  Pattern: {bold_pattern}")
    print(f"  Found {len(bold_files)} BOLD files")
    if bold_files:
        for f in bold_files:
            print(f"    Found: {f}")

    # Find corresponding brain masks
    brain_masks = []
    for bold_file in bold_files:
        # Extract the full task portion (including run if present)
        filename = os.path.basename(bold_file)
        # Replace desc-preproc_bold with desc-brain_mask
        mask_filename = filename.replace("desc-preproc_bold", "desc-brain_mask")

        # Get the directory path
        bold_dir = os.path.dirname(bold_file)
        brain_mask_file = f"{bold_dir}/{mask_filename}"

        if os.path.exists(brain_mask_file):
            brain_masks.append(brain_mask_file)
            print(f"    Found mask: {mask_filename}")
        else:
            print(f"    WARNING: Missing mask: {mask_filename}")
            brain_masks.append(None)

    # return bold_files, brain_masks BOLD files across all sessions
    if story == "wheretheressmoke":
        bold_pattern = f"{base_path}/{subject}/ses-*/func/{subject}_ses-*_task-{story}_*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    else:
        bold_pattern = f"{base_path}/{subject}/ses-*/func/{subject}_ses-*_task-{story}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    bold_files = glob(bold_pattern)
    print("these are bold files")
    print(bold_files)

    # Find corresponding brain masks
    brain_masks = []
    for bold_file in bold_files:
        # Extract session from bold file path
        ses = bold_file.split("ses-")[1].split("/")[0]
        if story == "wheretheressmoke":
            brain_mask_pattern = f"{base_path}/{subject}/ses-{ses}/func/{subject}_ses-{ses}_task-{story}_*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        else:
            brain_mask_pattern = f"{base_path}/{subject}/ses-{ses}/func/{subject}_ses-{ses}_task-{story}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        brain_mask_files = glob(brain_mask_pattern)
        if brain_mask_files:
            brain_masks.append(brain_mask_files[0])
        else:
            print(f"Warning: No brain mask found for {story} session {ses}")
            brain_masks.append(None)

    return bold_files, brain_masks


# Load gray matter mask (should be consistent across sessions)
gm_mask_path = f"{base_path}/{subject}/ses-1/anat/{subject}_ses-1_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz"
print(f"Loading GM mask from: {gm_mask_path}")
graymask_nii = nib.load(gm_mask_path)
graymask_data = graymask_nii.get_fdata()
gm_mask = graymask_data > 0.5

# Step 1: Create union brain mask across all sessions
print("Creating union brain mask across all sessions...")
union_brain_mask = None
all_brain_masks = []

# Collect all brain masks
for story in stories:
    bold_files, brain_mask_files = find_story_files(base_path, subject, story)
    for brain_mask_file in brain_mask_files:
        if brain_mask_file is not None:
            brain_mask_data = nib.load(brain_mask_file).get_fdata().astype(bool)
            all_brain_masks.append(brain_mask_data)

            if union_brain_mask is None:
                union_brain_mask = brain_mask_data.copy()
            else:
                union_brain_mask = union_brain_mask | brain_mask_data

print(f"Union brain mask contains {union_brain_mask.sum()} voxels")

# Step 2: Combine with GM mask
final_mask = union_brain_mask & gm_mask
print(f"Final mask (brain + GM) contains {final_mask.sum()} voxels")

# Initialize surface projector
print("Initializing surface projector...")
surface_projector = SurfaceProjector()

# Step 3: Extract time series for each story (both volume and surface)
story_data_volume = {}  # Original volumetric data
story_data_surface = {}  # New surface data
missing_stories = []

# Special handling for test story (wheretheressmoke - repeated across sessions)
test_story = "wheretheressmoke"
test_story_data_volume = []
test_story_data_surface = []


def log_save(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[INFO] Saving processed NIfTI: {args[2]}")
        return func(*args, **kwargs)

    return wrapper


class ProcessedNiftiSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Output directory already exists: {output_dir}")

    @log_save
    def save(self, processed_data, affine, header, filename):
        output_path = os.path.join(self.output_dir, filename)
        processed_nii = nib.Nifti1Image(processed_data, affine, header)
        nib.save(processed_nii, output_path)
        return output_path


# Create output directory for processed NIfTI files
output_dir = f"noslice_{subject}_processed_nifti"
processed_saver = ProcessedNiftiSaver(output_dir)

for story in stories:
    bold_files, brain_mask_files = find_story_files(base_path, subject, story)
    print(bold_files)

    if not bold_files:
        print(f"  Warning: No BOLD file found for {story}")
        missing_stories.append(story)
        continue

    # Handle test story specially (collect all repetitions)
    if story == test_story:
        print(f"  Found {len(bold_files)} repetitions of test story")
        test_runs_4d = []  # Store 4D data for averaging

        for i, bold_file in enumerate(bold_files):
            print(f"    Loading repetition {i+1}: {bold_file}")
            try:
                bold_nii = nib.load(bold_file)
                bold_data = bold_nii.get_fdata()

                # Trim first 10 and last 10 timepoints from 4D data
                if bold_data.shape[3] > 20:
                    trimmed_bold_data = bold_data[:, :, :, 10:-10]
                    print(f"      4D shape after trimming: {trimmed_bold_data.shape}")
                else:
                    trimmed_bold_data = bold_data
                    print(
                        f"      Warning: Run too short to trim ({bold_data.shape[3]} timepoints)"
                    )

                test_runs_4d.append(trimmed_bold_data)

                # Extract volumetric masked time series for this run
                masked_timeseries = trimmed_bold_data[final_mask, :].T
                test_story_data_volume.append(masked_timeseries)

                print(f"      Volume shape: {masked_timeseries.shape}")

            except Exception as e:
                print(f"      Error loading repetition {i+1}: {e}")

        if test_story_data_volume and test_runs_4d:
            # Average volumetric data across repetitions
            averaged_test_data_volume = np.mean(test_story_data_volume, axis=0)
            story_data_volume[test_story] = averaged_test_data_volume

            # Average 4D data first, then project to surface
            averaged_4d_data = np.mean(test_runs_4d, axis=0)
            print(f"  Averaged 4D data shape: {averaged_4d_data.shape}")

            # Project averaged volume to surface (single projection)
            print(f"  Projecting averaged volume to surface...")
            surface_data = surface_projector.project_volume_to_surface(
                averaged_4d_data, bold_nii.affine
            )
            story_data_surface[test_story] = surface_data.combined

            # Save processed NIfTI
            output_filename = f"{subject}_task-{story}_space-MNI152NLin2009cAsym_res-2_desc-processed_bold.nii.gz"
            processed_saver.save(
                averaged_4d_data, bold_nii.affine, bold_nii.header, output_filename
            )

            print(f"  Averaged {len(test_story_data_volume)} repetitions")
            print(f"  Volume final shape: {averaged_test_data_volume.shape}")
            print(f"  Surface final shape: {surface_data.combined.shape}")

    else:
        # Handle regular stories (single occurrence)
        bold_file = bold_files[0]
        print(f"  Loading: {bold_file}")

        try:
            bold_nii = nib.load(bold_file)
            bold_data = bold_nii.get_fdata()

            # Trim first 10 and last 10 timepoints from 4D data
            original_shape = bold_data.shape
            if bold_data.shape[3] > 20:
                trimmed_bold_data = bold_data[:, :, :, 10:-10]
                print(f"  4D shape before trimming: {original_shape}")
                print(f"  4D shape after trimming: {trimmed_bold_data.shape}")
            else:
                trimmed_bold_data = bold_data
                print(
                    f"  Warning: Story too short to trim ({original_shape[3]} timepoints)"
                )

            # Extract volumetric masked time series
            masked_timeseries = trimmed_bold_data[final_mask, :].T  # (time, voxels)
            story_data_volume[story] = masked_timeseries

            # Project to surface
            print(f"  Projecting {story} to surface...")
            surface_data = surface_projector.project_volume_to_surface(
                trimmed_bold_data, bold_nii.affine
            )
            # Store as (timepoints, vertices)
            story_data_surface[story] = surface_data.combined

            # Save processed 4D NIfTI file
            output_filename = f"{subject}_task-{story}_space-MNI152NLin2009cAsym_res-2_desc-processed_bold.nii.gz"
            processed_saver.save(
                trimmed_bold_data, bold_nii.affine, bold_nii.header, output_filename
            )

            print(f"  Volume shape: {masked_timeseries.shape}")
            print(f"  Surface shape: {surface_data.combined.shape}")

        except Exception as e:
            print(f"  Error loading {story}: {e}")
            missing_stories.append(story)

# Summary
print(f"\nSuccessfully loaded {len(story_data_volume)} stories:")
print("Volume data:")
for story, data in story_data_volume.items():
    print(f"  {story}: {data.shape}")

print("Surface data:")
for story, data in story_data_surface.items():
    print(f"  {story}: {data.shape}")

if missing_stories:
    print(f"\nMissing stories ({len(missing_stories)}):")
    for story in missing_stories:
        print(f"  {story}")

# Verify all stories have same number of voxels/vertices
volume_voxel_counts = [data.shape[1] for data in story_data_volume.values()]
surface_vertex_counts = [data.shape[1] for data in story_data_surface.values()]

if len(set(volume_voxel_counts)) == 1:
    print(
        f"\n All volume stories have consistent voxel count: {volume_voxel_counts[0]}"
    )
else:
    print(f"\n⚠ Warning: Inconsistent volume voxel counts: {set(volume_voxel_counts)}")

if len(set(surface_vertex_counts)) == 1:
    print(
        f" All surface stories have consistent vertex count: {surface_vertex_counts[0]}"
    )
else:
    print(
        f" Warning: Inconsistent surface vertex counts: {set(surface_vertex_counts)}"
    )

print(f"\nFinal data structure:")
print(
    f"  Volume: dictionary with {len(story_data_volume)} stories, each (timepoints, {final_mask.sum()} voxels)"
)
print(
    f"  Surface: dictionary with {len(story_data_surface)} stories, each (timepoints, {surface_vertex_counts[0] if surface_vertex_counts else 'N/A'} vertices)"
)

# Special note about test story
if test_story in story_data_volume:
    print(
        f"\nTest story '{test_story}' was averaged across {len(test_story_data_volume)} repetitions"
    )
    print(
        f"This increases SNR and provides better model evaluation as described in the paper"
    )

# Save the volume data as pickle (original filename)
volume_output_file = f"noslice_{subject}_story_data.pkl"
print(f"\nSaving volume data to: {volume_output_file}")
with open(volume_output_file, "wb") as f:
    pickle.dump(story_data_volume, f)

# Save the surface data as pickle (new filename)
surface_output_file = f"noslice_{subject}_story_data_surface.pkl"
print(f"Saving surface data to: {surface_output_file}")
with open(surface_output_file, "wb") as f:
    pickle.dump(story_data_surface, f)

# Also save the final mask for reference
mask_file = f"noslice_{subject}_final_mask.nii.gz"
print(f"Saving final mask to: {mask_file}")
final_mask_nii = nib.Nifti1Image(
    final_mask.astype(np.uint8), graymask_nii.affine, graymask_nii.header
)
nib.save(final_mask_nii, mask_file)

print(f"\n Data saved successfully!")
print(f"  Volume story data: {volume_output_file}")
print(f"  Surface story data: {surface_output_file}")
print(f"  Final mask: {mask_file}")
print(f"  Processed NIfTI files: {output_dir}/")

# Show how to load the data back
print(f"\nTo load the data later:")
print(f"  # Volume data")
print(f"  with open('{volume_output_file}', 'rb') as f:")
print(f"      story_data_volume = pickle.load(f)")
print(f"  ")
print(f"  # Surface data")
print(f"  with open('{surface_output_file}', 'rb') as f:")
print(f"      story_data_surface = pickle.load(f)")
print(f"  ")
print(f"  # Access specific story")
print(f"  volume_timeseries = story_data_volume['avatar']")
print(f"  surface_timeseries = story_data_surface['avatar']")
print(f"  test_story_avg_volume = story_data_volume['{test_story}']")
print(f"  test_story_avg_surface = story_data_surface['{test_story}']")
