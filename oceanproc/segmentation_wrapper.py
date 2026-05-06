#!/usr/bin/env python3

from pathlib import Path
import logging
from sre_constants import SUCCESS
from .utils import exit_program_early, make_option, prepare_subprocess_logging, flags, debug_logging, log_linebreak, run_subprocess, update_permissions
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

segmentation_args = {
    "start",
    "starting-stage",
    "end",
    "ending-stage",
    "fsl-bin-path",
    "overwrite-old", 
    "overwrite"
}

@debug_logging
def run_bibsnet(subject:str,
                session:str,
                bids_path:Path,
                derivs_path:Path,
                work_path:Path,
                bibsnet_image:Path,
                option_chain:str,
                remove_work_folder:bool = False):
    """
    Run BIBSnet with parameters.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param session: Name of session (ex. if path contains 'ses-01', session='01')
    :type session: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param work_path: Path to the working directory
    :type work_path: pathlib.Path
    :param bibsnet_image: Path to the bibsnet '.sif' apptainer image
    :type bibsnet_image: pathlib.Path
    :param option_chain: String containing generated list of options built by make_option().
    :type option_chain: str
    :param remove_work_folder: Flag to indicate if the work_path should be deleted (default False)
    :type remove_work_folder: bool
    :raise RuntimeError: If BIBSnet throws an error, or exits with a non-zero exit code.
    """
    def clean_up(): return shutil.rmtree(work_path) if remove_work_folder else None

    log_linebreak()
    logger.info(f"####### Starting BIBSnet #######\n")
    if not bids_path.exists():
        exit_program_early(f"Bids path {bids_path} does not exist.")
    elif not derivs_path.exists():
        exit_program_early(f"Derivatives path {derivs_path} does not exist.")
    elif not bibsnet_image.exists():
        exit_program_early(f"BIBSnet image path {derivs_path} does not exist.")

    bids_mount = "/data"
    derivs_mount = "/out"
    work_mount = "/work"
    bibsnet_command = f"""apptainer run --nv --cleanenv --no-home
                            -B {bids_path}:{bids_mount}:ro
                            -B {derivs_path}:{derivs_mount}
                            -B {work_path}:{work_mount}
                            {bibsnet_image} {bids_mount} {derivs_mount} participant
                            --participant-label {subject} 
                            --session-id {session}
                            -w {work_mount} --verbose
                            {option_chain}"""
    
    success = True
    try:
        run_subprocess(bibsnet_command, title="BIBSnet")
    except (Exception, KeyboardInterrupt) as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        success = False

    if not flags.debug:
        clean_up()

    return success

@debug_logging
def segment_anatomical(subject:str,
                 session:str,
                 bids_path:Path,
                 derivs_path:Path,
                 work_path:Path,
                 bibsnet_image:Path,
                 remove_work_folder:bool,
                 **kwargs):
    """
    Faciliates the running of BIBSnet.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param work_path: Path to the working directory that will be deleted upon completion or error
    :type work_path: Path
    :param bibsnet_image: Path to the bibsnet '.sif' apptainer image
    :type bibsnet_image: pathlib.Path
    :param remove_work_folder: Path to the working directory that will be deleted upon completion or error (default None)
    :type remove_work_folder: str
    :param **kwargs: any arguments to be passed to the fmriprep subprocess
    """
    option_chain = " ".join([make_option(v, key=k, delimeter=" ", convert_underscore=True) for k,v in kwargs.items()])

    segment_success = run_bibsnet(subject=subject,
                      session=session,
                      bids_path=bids_path,
                      derivs_path=derivs_path,
                      work_path=work_path,
                      bibsnet_image=bibsnet_image,
                      option_chain=option_chain,
                      remove_work_folder=remove_work_folder)
    
    paths_to_update = sorted(derivs_path.glob(f"bibsnet/sub-{subject}*"))
    paths_to_update.extend(sorted(work_path.glob(f"*/sub-{subject}*")))
    for out_path in paths_to_update:
        if out_path.exists():
            # update file permissions for segmentation outputs and working files
            update_permissions(
                permissions=flags.file_permissions, 
                path=out_path, 
                recursive=True,
                group=flags.permissions_group)
        
    # check other top-level files
    other_paths = [derivs_path/"dataset_description.json"]
    for op in other_paths:
        if op.exists() and (op.stat().st_uid == flags.uid):
            update_permissions(
                permissions=flags.file_permissions,
                path=op,
                recursive=True,
                group=flags.permissions_group
            )
        
    if not segment_success:
        exit_program_early(f"Program 'BIBSnet' has run into an error.")


