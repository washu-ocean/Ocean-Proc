#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import logging
import datetime
from .bids_wrapper import dicom_to_bids, remove_unusable_runs, extend_session
from .group_series import map_fmap_to_func, map_fmap_to_func_with_pairing_file
from .preprocessing_wrapper import process_data, adult_defaults, infant_defaults, mount_opts
from .segmentation_wrapper import segmentation_args, segment_anatomical
from .events_long import create_events_and_confounds
from .utils import exit_program_early, prompt_user_continue, default_log_format, add_file_handler, export_args_to_file, flags, debug_logging, log_linebreak, extract_options, make_option
from .oceanparse import OceanParser
import shutil
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
        epilog="An arguments file can be accepted with @FILEPATH.\nAny additional arguments not listed in the help message will be passed to the preprocessing subprocess.",
        allow_abbrev=False
    )
    parser.add_argument("--version", action="version", version='%(prog)s v1.1.2')
    session_args = parser.add_argument_group("Session Specific")
    config_args = parser.add_argument_group("Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")
    parser.register("type", "Full-Path", lambda p: Path(p).resolve())

    session_args.add_argument("--subject", "-su", required=True,
                              help="The identifier of the subject to preprocess")
    session_args.add_argument("--session", "-se", required=True,
                              help="The identifier of the session to preprocess")
    session_args.add_argument("--source_data", "-sd", type=Path,
                              help="The path to the directory containing the raw DICOM files for this subject and session")
    session_args.add_argument("--skip_dcm2bids", action="store_true",
                              help="Flag to indicate that dcm2bids does not need to be run for this subject and session")
    session_args.add_argument("--usability_file", "-u", type=Path,
                              help="The path to the usability file for this subject and session; this file can be either be an xml or json file. All runs will be used if no file is provided.")
    session_args.add_argument("--longitudinal", "-lg", type=Path, nargs="+", action="append",
                              help="Addtional sourcedata and usability pair to include with this BIDs subject and session")
    session_args.add_argument("--skip_fmap_pairing", action="store_true",
                              help="Flag to indicate that the pairing of fieldmaps to BOLD runs does not need to be performed for this subject and session")
    session_args.add_argument("--skip_segmentation", action="store_true",
                              help="Flag to indicate that segmentation does not need to be run for this subject and session")
    session_args.add_argument("--skip_preproc", action="store_true",
                              help="Flag to indicate that preprocessing does not need to be run for this subject and session")
    session_args.add_argument("--skip_event_files", action="store_true",
                              help="Flag to indicate that the making of a long formatted events file is not needed for the subject and session")
    session_args.add_argument("--export_args", "-ea", type=Path,
                              help="Path to a file to save the current configuration arguments")
    session_args.add_argument("--keep_work_dir", action="store_true",
                              help="Flag to stop the deletion of the fMRIPrep working directory")
    session_args.add_argument("--debug_mode", action="store_true",
                              help="Flag to run the program in debug mode for more verbose logging")
    session_args.add_argument("--fmap_pairing_file", type=Path,
                              help="Path to JSON containing info on how to pair fieldmaps to BOLD runs.")

    config_args.add_argument("--bids_path", "-b", type=Path, required=True,
                             help="The path to the directory containing the raw nifti data for all subjects, in BIDS format")
    config_args.add_argument("--derivs_path", "-d", type=Path, required=True,
                             help="The path to the BIDS formated derivatives directory for this subject")
    config_args.add_argument("--derivs_subfolder", "-ds", default=None,
                             help=f"""The subfolder in the derivatives directory where bids style outputs should be stored. 
                                  The default is {adult_defaults.derivs_subfolder} or {infant_defaults.derivs_subfolder} if the '--infant' flag is set.""")
    config_args.add_argument("--bids_config", "-c", type=Path, required=True,
                             help="The path to the dcm2bids config file to use for this subject and session")
    config_args.add_argument("--nordic_config", "-n", type=Path,
                             help="The path to the second dcm2bids config file to use for this subject and session. This implies that the session contains NORDIC data")
    config_args.add_argument("--nifti", action=argparse.BooleanOptionalAction,
                             help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    config_args.add_argument("--anat_only", action=argparse.BooleanOptionalAction,
                             help="Flag to specify only anatomical images should be processed.")
    config_args.add_argument("--fd_spike_threshold", "-fd", type=float, default=0.9,
                             help="framewise displacement threshold (in mm) to determine outlier framee (Default is 0.9).")
    config_args.add_argument("--skip_bids_validation", action=argparse.BooleanOptionalAction,
                             help="Specifies skipping BIDS validation (only enabled for fMRIprep step)")
    config_args.add_argument("--fs_subjects_dir", "-fs", type=Path,
                             help="The path to the directory that contains previous FreeSurfer outputs/derivatives to use for fMRIPrep. If empty, this is the path where new FreeSurfer outputs will be stored.")
    config_args.add_argument("--allow_uneven_fmap_groups", action="store_true",
                             help="Flag to allow for automated fieldmap pairing when there's more AP- than PA-encoded fieldmaps, or vice versa (will still error out if at least one of each is not present.)")
    config_args.add_argument("--precomputed_derivatives", "-pd", type="Full-Path", dest="derivatives", nargs="*",
                             help="A list of paths to any BIDS-style precomputed derivatives that should be used in preprocessing. (Ex. /path/to/bibsnet)")
    config_args.add_argument("--work_dir", "-w", type=Path, required=True,
                             help="The path to the working directory used to store intermediate files")
    config_args.add_argument("--fs_license", "-l", type=Path, required=True,
                             help="The path to the license file for the local installation of FreeSurfer")
    config_args.add_argument("--image_version", "-iv", default=None,
                             help=f"""The version of fmriprep to use; It is reccomended that an entire study use the same version. 
                                  The default is {adult_defaults.image_version} or {infant_defaults.image_version} if the '--infant' flag is set.""")
    config_args.add_argument("--infant", "-I", action=argparse.BooleanOptionalAction,
                             help="Flag to specify that NiBabies should be used instead of fMRIPrep")
    config_args.add_argument("--bibsnet_image_path", "-bi", type=Path,
                             help="Path to the BIBSnet apptainer image to use for segmentation. If provided, BIBSnet segmentation will be run and the outputs will be used for preprocessing. (Must be used with the '--infant' flag)")
    config_args.add_argument("--bibsnet_work", "-bw", type=Path,
                             help="The path to the working directory used to store intermediate files for BIBSnet")
    args, unknown_args = parser.parse_known_args()

    try:
        assert args.derivs_path.is_dir(), "Derivatives directory must exist but it cannot be found"
        assert args.bids_path.is_dir(), "Raw Bids directory must exist but it cannot be found"
        if not args.skip_dcm2bids:
            assert args.source_data.is_dir(), "Source data directory must exist but it cannot be found"
        assert args.work_dir.is_dir(), "Work directroy must exist but it cannot be found"
    except AssertionError as e:
        logger.exception(e)
        parser.error(e)

    defaults = infant_defaults if args.infant else adult_defaults
    for k,v in defaults.__dict__.items():
        if k in args.__dict__ and args.__dict__[k] is None:
            args.__dict__[k] = v

    unknown_args = extract_options(unknown_args)

    bibsnet_path = (args.derivs_path / "bibsnet").resolve()
    if args.infant and args.bibsnet_image_path:
        if args.derivatives is None:
            args.derivatives = [bibsnet_path]
        elif bibsnet_path not in args.derivatives:
            args.derivatives.append(bibsnet_path)

    if args.bibsnet_work and (not args.bibsnet_work.exists()):
        parser.error("BIBSnet working directory must exist but it cannot be found")

    if args.longitudinal:
        for lg_group in args.longitudinal:
            if not lg_group[0].is_dir():
                parser.error(f"Cannot find the souredata directory at the path: {lg_group[0]}")
            if len(lg_group) == 1:
                lg_group.append(None)
            elif len(lg_group) == 2:
                if not (lg_group[1].suffix in [".xml", ".json"]) or not lg_group[1].is_file():
                    parser.error(f"usability file must be a .xml or .json file, and it must exist: {lg_group[1]}")
            else:
                parser.error(f"longitudinal argument cannot have more than 2 elements: {lg_group}")

    preproc_image = f"{defaults.image_name}:{args.image_version}"
    preproc_derivs_path = args.derivs_path / args.derivs_subfolder

    log_dir = preproc_derivs_path / f"sub-{args.subject}/log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{args.subject}_ses-{args.session}_oceanproc_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}.log"
    add_file_handler(logger, log_path)
    if args.debug_mode:
        flags.debug = True
        logger.setLevel(logging.DEBUG)

    logger.info("Starting oceanproc...")
    logger.info(f"Log will be stored at {log_path}")

    ##### Export the current configuration arguments to a file #####
    if args.export_args:
        try:
            assert args.export_args.parent.exists() and args.export_args.suffix, "Argument export path must be a file path in a directory that exists"
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######")
            export_args_to_file(args, config_arguments, args.export_args, extra_args=unknown_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    if args.longitudinal:
        flags.longitudinal = True
    if args.infant:
        flags.infant = True

    # log the input arguments
    for k,v in (dict(args._get_kwargs())).items():
        logger.info(f" {k} : {v}")
    logger.info(f"extra arguments: {unknown_args}")

    preproc_work_dir = make_work_directory(dir_path=args.work_dir,
                                           subject=args.subject,
                                           session=args.session)
    bibsnet_work_dir = args.bibsnet_work if args.bibsnet_work else args.work_dir

    dicom_sessions = [(args.source_data, args.usability_file, args.bids_path)]
    if flags.longitudinal:
        for soure_dir, use_file in args.longitudinal:
            dicom_sessions.append((soure_dir, use_file, preproc_work_dir))

    for index, (souredata_path, use_file, bids_path) in enumerate(dicom_sessions):
        ##### Convert raw DICOMs to BIDS structure #####
        if not args.skip_dcm2bids:
            dicom_to_bids(
                subject=args.subject,
                session=args.session,
                source_dir=souredata_path,
                bids_dir=bids_path,
                usability_file=use_file,
                bids_config=args.bids_config,
                nordic_config=args.nordic_config,
                nifti=args.nifti,
                skip_validate=args.skip_bids_validation,
                skip_prompt=index > 0,
                session_index=index
            )
            if index > 0:
                extend_session(subject=args.subject,
                               session=args.session,
                               tmp_dir=bids_path,
                               bids_dir=args.bids_path)

    ##### Remove the scans marked as 'unusable' #####
    remove_unusable_runs(
        bids_path=args.bids_path,
        subject=args.subject,
        session=args.session
    )

    ##### Pair field maps to functional runs #####
    if not args.anat_only and not args.skip_fmap_pairing:
        if args.fmap_pairing_file:
            bids_session_dir = args.bids_path / f"sub-{args.subject}/ses-{args.session}"
            map_fmap_to_func_with_pairing_file(
                bids_session_dir,
                args.fmap_pairing_file
            )
        else:
            map_fmap_to_func(
                subject=args.subject,
                session=args.session,
                bids_path=args.bids_path,
                # xml_path=xml_data_path,
                allow_uneven_fmap_groups=args.allow_uneven_fmap_groups
            )

    ##### Run BIBSnet #####
    bibsnet_args = {key:unknown_args[key] for key in segmentation_args if key in unknown_args}
    for key in segmentation_args:
        if key in unknown_args:
            del unknown_args[key]

    if flags.infant and args.bibsnet_image_path and (not args.skip_segmentation):
        segment_anatomical(
            subject=args.subject,
            session=args.session,
            bids_path=args.bids_path,
            derivs_path=args.derivs_path,
            work_path=bibsnet_work_dir,
            bibsnet_image=args.bibsnet_image_path,
            remove_work_folder=False,
            **bibsnet_args
        )

        if not bibsnet_path.exists():
            exit_program_early(f"Cannot find the outputs for BIBSnet at the path {bibsnet_path}")

    ##### Run fMRIPrep or NiBabies #####
    all_opts = dict(args._get_kwargs())

    additional_mounts = {make_option(True, mo, convert_underscore=True).strip():all_opts[mo] for mo in mount_opts if all_opts[mo]}
    fmrip_options = {"skip_bids_validation",
                     "fd_spike_threshold",
                     "anat_only"}

    if not args.skip_preproc:
        process_data(
            subject=args.subject,
            session=args.session,
            bids_path=args.bids_path,
            derivs_path=preproc_derivs_path,
            work_path=preproc_work_dir,
            license_file=args.fs_license,
            image_name=preproc_image,
            additional_mounts=additional_mounts,
            remove_work_folder=not args.keep_work_dir,
            **({o:all_opts[o] for o in fmrip_options} | unknown_args)
        )

    ##### Create long formatted event files #####
    if not args.skip_event_files:
        create_events_and_confounds(
            bids_path=args.bids_path,
            derivs_path=preproc_derivs_path,
            sub=args.subject,
            ses=args.session,
            fd_thresh=args.fd_spike_threshold
        )

    log_linebreak()
    logger.info("####### [DONE] Finished all processing, exiting now #######")


if __name__ == "__main__":
    main()
