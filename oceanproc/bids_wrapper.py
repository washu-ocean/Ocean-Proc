#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from pathlib import Path
import json
import os
import re
from datetime import datetime
import shlex
import shutil
import subprocess
from textwrap import dedent
import xml.etree.ElementTree as et
from .utils import exit_program_early, prompt_user_continue, prepare_subprocess_logging, debug_logging, log_linebreak, flags
import logging

logger = logging.getLogger(__name__)
'''
remove_unusable -> relate xml and json with protocol name and aqcuisition time (and series id?)
'''
@debug_logging
def remove_unusable_runs(xml_file:Path, bids_data_path:Path, subject:str, session:str):
    """
    Will remove unusable scans from list of scans after dcm2bids has run.

    :param xml_file: Path to XML file containing quality information generated by XNAT.
    :type xml_file: pathlib.Path
    :param bids_data_path: Path to directory containing BIDS-compliant data for the given subject.
    :type bids_data_path: pathlib.Path
    :param subject: Subject ID used in BIDS-compliant data (for example, if 'sub-5000', subject is '5000').
    :type subject: str
    :param session: Session ID used in BIDS-compliant data (for example, if 'ses-01', session is '01').
    :type session: str

    """
    log_linebreak()
    logger.info("####### Removing the scans marked 'unusable' #######\n")

    if not xml_file.exists():
        exit_program_early(f"Path {str(xml_file)} does not exist.")
        
    tree = et.parse(xml_file)
    prefix = "{" + str(tree.getroot()).split("{")[-1].split("}")[0] + "}"
    scan_element_list = list(tree.iter(f"{prefix}scans"))
    
    if len(scan_element_list) != 1:
        exit_program_early(f"Error parsing the xml file provided. Found none or more than one scan groups")
    
    scans = scan_element_list[0]
    quality_pairs = {}
    for s in scans:
        series_id = int(re.split(r'[-,;_.]', s.get("ID"))[0])
        series_desc = s.find(f"{prefix}series_description").text
        protocal_name = s.find(f"{prefix}protocolName").text
        quality_info = s.find(f"{prefix}quality").text
        s_key = (series_id, series_desc, protocal_name)
        if s_key in quality_pairs and (quality_info=="unusable" or quality_pairs[s_key]=="unusable"):
            exit_program_early(f"Found scans with identical series numbers and protocol names in the session xml file. Cannot accurately differentiate these scans {s_key}")
        quality_pairs[s_key] = quality_info

    if len(quality_pairs) == 0:
        exit_program_early("Could not find scan quality information in the given xml file.") 

    logger.info(f"scan quality information: ")
    for k, v in quality_pairs.items():
        logger.info(f"\t{k} -> {v}")
    
    json_paths = sorted(list(p for p in (bids_data_path / f"sub-{subject}/ses-{session}").rglob("*.json"))) 

    if len(json_paths) == 0:
        exit_program_early("Could not find JSON sidecar files in the bids directory.")

    for p_json in json_paths:
        p_nii = p_json.with_suffix(".nii.gz")
        if not p_nii.exists():
            exit_program_early(f"Could not find the NIFTI file that goes with the sidecar file: {p_json} ")
        j = json.load(p_json.open()) 
        j_series_id, j_series_desc, j_protocol_name = j["SeriesNumber"], j["SeriesDescription"], j["ProtocolName"]
        j_key = (j_series_id, j_series_desc, j_protocol_name)
        if quality_pairs[j_key] == "unusable":
            logger.info(f"  Removing series {j_series_id} - {j_series_desc} - {j_protocol_name}: \n\t NIFTI:{p_nii}, JSON:{p_json}")
            os.remove(p_json) 
            os.remove(p_nii) 


@debug_logging
def run_dcm2bids(subject:str, 
                 session:str,
                 nifti_dir:Path, 
                 bids_output_dir:Path, 
                 config_file:Path, 
                 nordic_config:Path=None):
    """
    Run dcm2bids with a given set of parameters.

    :param subject: Subject name (ex. 'sub-5000', subject would be '5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param nifti_dir: Path to the directory where NIFTI data for this session is kept.
    :type nifti_dir: pathlib.Path
    :param bids_output_dir: Path to the bids directory to store the newly made NIFTI files
    :type bids_output_dir: pathlib.Path
    :param config_file: Path to dcm2bids config file, which maps raw sourcedata to BIDS-compliant counterpart
    :type config_file: pathlib.Path
    :param nordic_config: Path to second dcm2bids config file, needed for additional post processing that one BIDS config file can't handle.
    :type nordic_config: pathlib.Path
    :raise RuntimeError: If dcm2bids exits with a non-zero exit code.
    """

    if shutil.which('dcm2bids') == None:
            exit_program_early("Cannot locate program 'dcm2bids', make sure it is in your PATH.")

    helper_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                 --bids_validate 
                                 --skip_dcm2niix
                                 -d {str(nifti_dir)} 
                                 -p {subject} 
                                 -s {session} 
                                 -c {str(config_file)} 
                                 -o {str(bids_output_dir)}
                                 """)
    try:
        log_linebreak()
        logger.info("####### Running first round of Dcm2Bids ########\n")
        prepare_subprocess_logging(logger)
        with subprocess.Popen(helper_command, stdout=subprocess.PIPE) as p:    
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
            
        if nordic_config:
            if not nordic_config.exists():
                exit_program_early(f"Path {nordic_config} does not exist.")

            nordic_run_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                            --bids_validate
                                            --skip_dcm2niix
                                            -d {str(nifti_dir)} 
                                            -p {subject}
                                            -s {session}
                                            -c {str(nordic_config)}
                                            -o {str(bids_output_dir)}
                                            """)
            log_linebreak()
            logger.info("####### Running second round of Dcm2Bids ########\n")
            prepare_subprocess_logging(logger)
            with subprocess.Popen(nordic_run_command, stdout=subprocess.PIPE) as p:
                while p.poll() == None:
                    for line in p.stdout:
                        logger.info(line.decode("utf-8", "ignore"))
                prepare_subprocess_logging(logger, stop=True)
                p.kill()
                if p.poll() != 0:
                    raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
                
            # Clean up NORDIC files
            separate_nordic_files = glob(f"{str(bids_output_dir)}/sub-{subject}/ses-{session}/func/*_part-*")
            logger.debug(f"removing the old nordic files that are not needed after mag-phase combination :")
            for f in separate_nordic_files:
                logger.debug(f"deleting file: {f}")
                os.remove(f)

    except RuntimeError or subprocess.CalledProcessError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early("Problem running 'dcm2bids'.")


@debug_logging
def run_dcm2niix(source_dir:Path, 
                 tmp_nifti_dir:Path,
                 clean_up_func=None):
    """
    Run dcm2niix with the given input and output directories.

    :param source_dir: Path to 'sourcedata' directory (or wherever DICOM data is kept).
    :type source_dir: pathlib.Path
    :param tmp_nifti_dir: Path to the directory to store the newly made NIFTI files
    :type tmp_nifti_dir: pathlib.Path
    """
    
    if not source_dir.exists():
        exit_program_early(f"Path {source_dir} does not exist.")
    elif shutil.which('dcm2niix') == None:
        exit_program_early("Cannot locate program 'dcm2niix', make sure it is in your PATH.")

    if not tmp_nifti_dir.exists():
        tmp_nifti_dir.mkdir(parents=True)
    
    helper_command = shlex.split(f"""{shutil.which('dcm2niix')} 
                                -b y
                                -ba y
                                -z y
                                -f %3s_%f_%p_%t
                                -o {str(tmp_nifti_dir)}
                                {str(source_dir)}
                                """)
    try: 
        log_linebreak()
        logger.info("####### Converting DICOM files into NIFTI #######\n")
        prepare_subprocess_logging(logger)
        with subprocess.Popen(helper_command, stdout=subprocess.PIPE) as p:
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'dcm2bniix' has ended with a non-zero exit code.")
    except RuntimeError or subprocess.CalledProcessError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early("Problem running 'dcm2niix'.", 
                           exit_func=clean_up_func if clean_up_func else None)
        
    # Delete or move extra files from short runs
    files_to_remove = list(tmp_nifti_dir.glob("*a.nii.gz")) + list(tmp_nifti_dir.glob("*a.json"))
    if flags.debug:
        unused_files_dir = tmp_nifti_dir.parent / f"{tmp_nifti_dir.name}_unused"
        unused_files_dir.mkdir(exist_ok=True)
        logger.debug(f"moving some unused files and files created from shortened runs to directory {unused_files_dir}")
        for f in files_to_remove:
            shutil.move(f.resolve(), (unused_files_dir/f.name).resolve())
    else:
        logger.info(f"removing some unused files and files created from shortened runs :\n  {[str(f) for f in files_to_remove]}")
        for f in files_to_remove:
            os.remove(f)
    
    
@debug_logging
def dicom_to_bids(subject:str, 
                  session:str, 
                  source_dir:Path, 
                  bids_dir:Path, 
                  xml_path:Path, 
                  bids_config:Path,
                  nordic_config:Path=None,
                  nifti=False):
    
    """
    Facilitates the conversion of DICOM data into NIFTI data in BIDS format, and the removal of data marked 'unusable'.

    :param subject: Subject name (ex. 'sub-5000', subject would be '5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param source_dir: Path to 'sourcedata' directory (or wherever DICOM data is kept).
    :type source_dir: pathlib.Path
    :param bids_dir: Path to the bids directory to store the newly made NIFTI files
    :type bids_dir: pathlib.Path
    :param bids_config: Path to dcm2bids config file, which maps raw sourcedata to BIDS-compliant counterpart
    :type bids_config: pathlib.Path
    :param nordic_config: Path to second dcm2bids config file, needed for additional post processing if NORDIC data that one BIDS config file can't handle.
    :type nordic_config: pathlib.Path
    :param nifti: Specify that the soure directory contains NIFTI files instead of DICOM
    :type nifti: bool
    """

    for p in [source_dir, bids_dir, bids_config]:
        if not p.exists():
            exit_program_early(f"Path {str(p)} does not exist.")

    if (path_that_exists := bids_dir/f"sub-{subject}/ses-{session}").exists():
            ans = prompt_user_continue(dedent(f"""
                                        A raw data bids path for this subject and session already exists. 
                                        Would you like to delete its contents and rerun dcm2bids? If not,
                                        dcm2bids will be skipped.
                                            """))
            if ans:
                logger.debug("removing the old BIDS raw data directory and its contents")
                shutil.rmtree(path_that_exists)
            else:
                return
            
    tmp_path = bids_dir / f"tmp_dcm2bids/sub-{subject}_ses-{session}"
    def clean_up(quiet=False):
        try:
            logger.debug(f"removing the temporary directory used by dcm2bids: {tmp_path}")
            shutil.rmtree(tmp_path)
        except Exception:
            if not quiet:
                logger.warning(f"There was a problem deleting the temporary directory at {tmp_path}")
    
    nifti_path = None
    clean_up(quiet=True)    
    if not nifti:
        run_dcm2niix(source_dir=source_dir, 
                     tmp_nifti_dir=tmp_path)
        nifti_path = tmp_path
    else:
        nifti_path = source_dir
    

    run_dcm2bids(subject=subject, 
                 session=session,
                 nifti_dir=nifti_path, 
                 bids_output_dir=bids_dir, 
                 config_file=bids_config, 
                 nordic_config=nordic_config)
    
    if not flags.debug:
        clean_up()
        


if __name__ == "__main__":
    parser = ArgumentParser(prog="bids_wrapper.py",
                                    description="wrapper script for dcm2bids",
                                    epilog="WIP")
    parser.add_argument("-su", "--subject", required=True, 
                        help="Subject ID")
    parser.add_argument("-se","--session", required=True, 
                        help="Session ID")
    parser.add_argument("-sd", "--source_data", type=Path, required=True,
                        help="Path to directory containing this session's DICOM files")
    parser.add_argument("-b", "--bids_path", type=Path, required=True, 
                        help="Path to the bids directory to store the newly made NIFTI files")
    parser.add_argument("-x", "--xml_path", type=Path, required=True, 
                        help="Path to this session's XML file")
    parser.add_argument("-c", "--bids_config", type=Path, required=True, 
                        help="dcm2bids config json file")
    parser.add_argument("-n", "--nordic_config", type=Path,
                        help="Second dcm2bids config json file used for NORDIC processing")
    parser.add_argument("--nifti", action='store_true', 
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    args = parser.parse_args()
    
    dicom_to_bids(subject=args.subject,
                  session=args.session,
                  source_dir=args.source_data,
                  bids_dir=args.bids_path,
                  xml_path=args.xml_path,
                  bids_config=args.bids_config,
                  nordic_config=args.nordic_config,
                  nifti=args.nifti)
