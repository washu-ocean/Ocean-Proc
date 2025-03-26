#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import logging
import datetime
from .bids_wrapper import dicom_to_bids, remove_unusable_runs
from .group_series import map_fmap_to_func, map_fmap_to_func_with_pairing_file
from .fmriprep_wrapper import process_data
from .events_long import create_events_and_confounds
from .utils import exit_program_early, prompt_user_continue, default_log_format, add_file_handler, export_args_to_file, flags, debug_logging, log_linebreak
from .oceanparse import OceanParser
import shlex
import shutil
from subprocess import Popen, PIPE
from textwrap import dedent

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)],
                    format=default_log_format)
logger = logging.getLogger() 

@debug_logging
def make_work_directory(dir_path:Path, subject:str, session:str) -> Path:
    dir_to_make = dir_path / f"sub-{subject}_ses-{session}"
    if dir_to_make.exists():
        want_to_delete = prompt_user_continue(dedent("""
            A work directory already exists for this subject and session. 
            Would you like to delete its contents and start fresh?
            """))
        if want_to_delete:
            logger.debug("removing the old working directory and its contents")
            shutil.rmtree(dir_to_make)
        else:
            return dir_to_make
    dir_to_make.mkdir()
    logger.info(f"creating a new working directory at the path: {dir_to_make}")
    return dir_to_make


def main():
    parser = OceanParser(
            prog="oceanproc", 
            description="Ocean Labs adult MRI preprocessing",
            exit_on_error=False,
            fromfile_prefix_chars="@",
            epilog="An arguments file can be accepted with @FILEPATH"
        )
    session_arguments = parser.add_argument_group("Session Specific")
    config_arguments = parser.add_argument_group("Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")

    session_arguments.add_argument("--subject", "-su", required=True,
                        help="The identifier of the subject to preprocess")
    session_arguments.add_argument("--session", "-se", required=True,
                        help="The identifier of the session to preprocess")
    session_arguments.add_argument("--source_data", "-sd", type=Path, required=True,
                        help="The path to the directory containing the raw DICOM files for this subject and session")
    session_arguments.add_argument("--xml_path", "-x", type=Path, required=True,
                        help="The path to the xml file for this subject and session")
    session_arguments.add_argument("--skip_dcm2bids", action="store_true",
                        help="Flag to indicate that dcm2bids does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_fmap_pairing", action="store_true",
                        help="Flag to indicate that the pairing of fieldmaps to BOLD runs does not need to be performed for this subject and session")
    session_arguments.add_argument("--skip_fmriprep", action="store_true",
                        help="Flag to indicate that fMRIPrep does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_event_files", action="store_true",
                        help="Flag to indicate that the making of a long formatted events file is not needed for the subject and session")
    session_arguments.add_argument("--export_args", "-ea", type=Path,
                        help="Path to a file to save the current configuration arguments")
    session_arguments.add_argument("--keep_work_dir", action="store_true",
                        help="Flag to stop the deletion of the fMRIPrep working directory")
    session_arguments.add_argument("--debug", action="store_true",
                        help="Flag to run the program in debug mode for more verbose logging")
    session_arguments.add_argument("--fmap_pairing_file",
                                   help="Path to JSON containing info on how to pair fieldmaps to BOLD runs.",
                                   type=Path)

    config_arguments.add_argument("--bids_path", "-b", type=Path, required=True,
                        help="The path to the directory containing the raw nifti data for all subjects, in BIDS format") 
    config_arguments.add_argument("--derivs_path", "-d", type=Path, required=True,
                        help="The path to the BIDS formated derivatives directory for this subject")
    config_arguments.add_argument("--derivs_subfolder", "-ds", default="fmriprep",
                        help="The subfolder in the derivatives directory where bids style outputs should be stored. The default is 'fmriprep'.")
    config_arguments.add_argument("--bids_config", "-c", type=Path, required=True,
                        help="The path to the dcm2bids config file to use for this subject and session")
    config_arguments.add_argument("--nordic_config", "-n", type=Path,
                        help="The path to the second dcm2bids config file to use for this subject and session. This implies that the session contains NORDIC data")
    config_arguments.add_argument("--nifti", action=argparse.BooleanOptionalAction,
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    config_arguments.add_argument("--anat_only", action='store_true',
                        help="Flag to specify only anatomical images should be processed.")
    config_arguments.add_argument("--fd_spike_threshold", "-fd", type=float, default=0.9,
                        help="framewise displacement threshold (in mm) to determine outlier framee (Default is 0.9).")
    config_arguments.add_argument("--skip_bids_validation", action="store_true",
                        help="Specifies skipping BIDS validation (only enabled for fMRIprep step)")
    config_arguments.add_argument("--fs_subjects_dir", "-fs", type=Path,
                        help="The path to the directory that contains previous FreeSurfer outputs/derivatives to use for fMRIPrep. If empty, this is the path where new FreeSurfer outputs will be stored.")
    config_arguments.add_argument("--work_dir", "-w", type=Path, required=True,
                        help="The path to the working directory used to store intermediate files")
    config_arguments.add_argument("--fs_license", "-l", type=Path, required=True,
                        help="The path to the license file for the local installation of FreeSurfer")
    config_arguments.add_argument("--fmriprep_version", "-fv", default="23.1.4", dest="image",
                        help="The version of fmriprep to use. The default is 23.1.4. It is reccomended that an entire study use the same version.")
    args = parser.parse_args()

    try:
        assert args.derivs_path.is_dir(), "Derivatives directory must exist but it cannot be found"
        assert args.bids_path.is_dir(), "Raw Bids directory must exist but it cannot be found"
        assert args.source_data.is_dir(), "Source data directory must exist but it cannot be found"
        assert args.work_dir.is_dir(), "Work directroy must exist but it cannot be found"
    except AssertionError as e:
        logger.exception(e)
        exit_program_early(e)

    ##### Export the current configuration arguments to a file #####
    if args.export_args:
        try:
            assert args.export_args.parent.exists() and args.export_args.suffix, "Argument export path must be a file path in a directory that exists"
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######")
            export_args_to_file(args, config_arguments, args.export_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    args.image = f"nipreps/fmriprep:{args.image}"

    args.derivs_path = args.derivs_path / args.derivs_subfolder

    log_dir = args.derivs_path / f"sub-{args.subject}/log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{args.subject}_ses-{args.session}_oceanproc_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}.log"
    add_file_handler(logger, log_path)

    if args.debug:
        flags.debug = True
        logger.setLevel(logging.DEBUG)

    logger.info("Starting oceanproc...")
    logger.info(f"Log will be stored at {log_path}")

    # log the input arguments
    for k,v in (dict(args._get_kwargs())).items():
        logger.info(f" {k} : {v}")

    args.work_dir = make_work_directory(dir_path=args.work_dir, 
                                        subject=args.subject, 
                                        session=args.session)

    ##### Convert raw DICOMs to BIDS structure #####
    if not args.skip_dcm2bids:
        dicom_to_bids(
            subject=args.subject,
            session=args.session,
            source_dir=args.source_data,
            bids_dir=args.bids_path,
            xml_path=args.xml_path,
            bids_config=args.bids_config,
            nordic_config=args.nordic_config,
            nifti=args.nifti
        )
    
    ##### Remove the scans marked as 'unusable' #####
    remove_unusable_runs(
        xml_file=args.xml_path,
        bids_data_path=args.bids_path,
        subject=args.subject,
        session=args.session
    )

    ##### Pair field maps to functional runs #####
    bids_session_dir = args.bids_path / f"sub-{args.subject}/ses-{args.session}"

    if not args.anat_only and not args.skip_fmap_pairing:
        if args.fmap_pairing_file:
            map_fmap_to_func_with_pairing_file(
                bids_session_dir,
                args.fmap_pairing_file
            )
        else:
            map_fmap_to_func(
                xml_path=args.xml_path, 
                bids_dir_path=bids_session_dir
            )


    ##### Run fMRIPrep #####
    all_opts = dict(args._get_kwargs())
    fmrip_options = {"work_dir", 
                     "fs_license", 
                     "fs_subjects_dir", 
                     "skip_bids_validation", 
                     "fd_spike_threshold", 
                     "anat_only",
                     "image"}

    if not args.skip_fmriprep:
        process_data(
            subject=args.subject, 
            session=args.session,
            bids_path=args.bids_path,
            derivs_path=args.derivs_path,
            remove_work_folder=None if args.keep_work_dir else args.work_dir,
            **{o:all_opts[o] for o in fmrip_options}
        )

    
    ##### Create long formatted event files #####
    if not args.skip_event_files:
        create_events_and_confounds(
            bids_path=args.bids_path,
            derivs_path=args.derivs_path,
            sub=args.subject,
            ses=args.session,
            fd_thresh=args.fd_spike_threshold
        )
    
    log_linebreak()
    logger.info("####### [DONE] Finished all processing, exiting now #######")

    
if __name__ == "__main__":
    main()
