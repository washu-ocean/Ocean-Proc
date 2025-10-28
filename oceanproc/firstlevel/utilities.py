from pathlib import Path
import nibabel as nib
import nilearn.masking as nmask
import numpy as np
import bids
from bids.layout.utils import PaddedInt

cifti_files = [
    ".dtseries.nii",
    ".ptseries.nii",
    ".dscalar.nii",
    ".pscalar.nii"
]


# Grab BOLD files from the preprocessed outputs, and list the runs and file extension for each 'func_space' in the files. 
def parse_session_bold_files(layout:bids.BIDSLayout, subject:str, session:str, task:str):
    files = layout.get(subject=subject, session=session, task=task, suffix="bold", datatype="func", extension=[".nii",".nii.gz",".dtseries.nii"])
    space_run_dict = dict()
    for f in files:
        run = f.entities["run"] if "run" in f.entities else PaddedInt('01')
        space = f.entities["space"] if "space" in f.entities else "func"
        if space in space_run_dict:
            space_run_dict[space]["runs"].append(run)
        else:
            space_run_dict[space] = {}
            space_run_dict[space]["extension"] = f.entities["extension"]
            space_run_dict[space]["runs"] = [run]
    return space_run_dict


def load_data(func_file: bids.layout.BIDSFile,
              brain_mask: str|Path) -> np.ndarray:
    func_file = str(func_file.path)
    img = nib.load(func_file)
    if is_cifti_file(func_file):
        return img.get_fdata()
    elif is_nifti_file(func_file):
        if brain_mask:
            return nmask.apply_mask(img, brain_mask)
        else:
            raise RuntimeError("Volumetric data must also have an accompanying brain mask")
        

def is_cifti_file(file: str|Path) -> str|None:
    if isinstance(file, Path):
        file = str(file)
    suffix = [cf for cf in cifti_files if file.endswith(cf)]
    return suffix[0] if len(suffix) > 0 else None

def is_nifti_file(file: str|Path) -> str|None:
    if isinstance(file, Path):
        file = str(file)
    not_cifti = is_cifti_file(file) is None
    suffix = [nf for nf in [".nii.gz", ".nii"] if file.endswith(nf)]
    return suffix[0] if (len(suffix) > 0) and not_cifti else None