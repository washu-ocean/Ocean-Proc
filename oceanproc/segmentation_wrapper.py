#!/usr/bin/env python3

from pathlib import Path
import logging
from .utils import exit_program_early, make_option, prepare_subprocess_logging, flags, debug_logging, log_linebreak, run_subprocess
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

    # uid = Popen(["id", "-u"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    # gid = Popen(["id", "-g"], stdout=PIPE).stdout.read().decode("utf-8").strip()

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
    
    try:
        run_subprocess(bibsnet_command, title="BIBSnet")
    except RuntimeError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early(f"Program 'BIBSnet' has run into an error.",
                           None if flags.debug else clean_up)
    if not flags.debug:
        clean_up()


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

    run_bibsnet(subject=subject,
                      session=session,
                      bids_path=bids_path,
                      derivs_path=derivs_path,
                      work_path=work_path,
                      bibsnet_image=bibsnet_image,
                      option_chain=option_chain,
                      remove_work_folder=remove_work_folder)


