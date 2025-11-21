from pathlib import Path
import nibabel as nib
import nilearn.masking as nmask
import numpy as np
import bids
from bids.layout.utils import PaddedInt
from bids.layout import parse_file_entities
from nipype.utils.filemanip import split_filename
# from bids.layout.writing import build_path

cifti_files = [
    ".dtseries.nii",
    ".ptseries.nii",
    ".dscalar.nii",
    ".pscalar.nii"
]


# Grab BOLD files from the preprocessed outputs, and list the runs and file extension for each 'func_space' in the files. 
def parse_session_bold_files(layout:bids.BIDSLayout, subject:str, session:str, tasks:list[str]):
    files = layout.get(subject=subject, session=session, task=tasks, suffix="bold", datatype="func", extension=[".nii",".nii.gz",".dtseries.nii"])
    space_run_dict = dict()
    for f in files:
        run = f.entities["run"] if "run" in f.entities else PaddedInt('01')
        space = f.entities["space"] if "space" in f.entities else "func"
        task = f.entities["task"]
        if space in space_run_dict:
            if task in space_run_dict[space]["runs"]:
                space_run_dict[space]["runs"][task].append(run)
            else:
                space_run_dict[space]["runs"][task] = [run]
        else:
            space_run_dict[space] = {}
            space_run_dict[space]["extension"] = f.entities["extension"]
            space_run_dict[space]["runs"] = {task:[run]}
    return space_run_dict


def load_data(func_file: str|Path|bids.layout.BIDSFile,
              brain_mask: str|Path) -> np.ndarray:
    if isinstance(func_file, bids.layout.BIDSFile):
        func_file = str(func_file.path)
    elif not isinstance(func_file, str):
        func_file = str(func_file)
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


def create_image_like(data: np.ndarray,
                 source_header,
                 out_file: str|Path,
                 dscalar_axis:list[str] = None,
                 brain_mask: str = None):

    if brain_mask:
        data_img = nmask.unmask(data, brain_mask)
        nib.save(data_img, out_file)
        return

    if isinstance(source_header, str) or isinstance(source_header, Path):
        source_header = nib.load(source_header).header
    elif not isinstance(source_header, nib.cifti2.cifti2.Cifti2Header):
        raise ValueError("source_header must be one of the following types: [str, pathlib.Path, nibabel.cifti2.cifti2.Cifti2Header]")
    
    ax0 = ( 
        nib.cifti2.cifti2_axes.ScalarAxis(name=dscalar_axis) 
            ) if  dscalar_axis else (
        nib.cifti2.cifti2_axes.SeriesAxis(start=0, step=source_header.get_axis(0).step, size=data.shape[0]) 
        )
    
    data_img = nib.cifti2.cifti2.Cifti2Image(data, (ax0, source_header.get_axis(1)))
    nib.save(data_img, out_file)
    return    

def replace_entities(file:str, entities:dict):
    
    for entity, value in entities.items():
        file = replace_entity(file, entity, value)
    return file


def replace_entity(file:str, entity:str, value:str):
    if entity == "suffix":
        prefix, suffix = file.rsplit("_", 1)
        ext = suffix.split(".",1)[-1]
        return f"{prefix}_{value}.{ext}"
    
    if entity == "ext":
        return f"{file.split('.',1)[0]}{value}"
    
    if entity == "path":
        fname=Path(file).name
        if not value:
            return str(Path().resolve()/fname)
        else:
            return f"{value}/{fname}"
    
    entity_label = f"_{entity}-"
    if entity_label in file:
        prefix, suffix = file.split(entity_label, 1)
        suffix = suffix.split("_",1)[-1]
        if value is None:
            return f"{prefix}_{suffix}"
        return f"{prefix}{entity_label}{value}_{suffix}"

    return file