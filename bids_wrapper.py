from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as et


def exit_program_early(msg:str):
        print(f"---[ERROR]: {msg} \nExiting the program now...")
        sys.exit(1)
     

def prompt_user_continue(msg:str):
    user_continue = input(f"--- {msg} (press 'y' for yes, other input will exit the program) ---")
    if user_continue != 'y':
        exit_program_early("User-prompted termination.")


def remove_unusable_runs(xml_file:Path, bids_data_path:Path, subject:str):
    print("####### Removing the scans marked 'unusable' #######")

    if not xml_file.exists():
        exit_program_early(f"Path {xml_file} does not exist.")
        
    tree = et.parse(xml_file)
    prefix = "{" + str(tree.getroot()).split("{")[-1].split("}")[0] + "}"
    scans = tree.getroot().find(
        f"./{prefix}experiments/{prefix}experiment/{prefix}scans"
    )
    quality_pairs = {int(s.get("ID")) : s.find(f"{prefix}quality").text
                     for s in scans}
    
    if len(quality_pairs) == 0:
        exit_program_early("Could not find scan quality information in the given xml file.") 
    
    json_paths = sorted(list(p for p in (bids_data_path / f"sub-{subject}/").rglob("*.json"))) # if re.search(json_re, p.as_posix()) != None)
    nii_paths = sorted(list(p for p in (bids_data_path / f"sub-{subject}/").rglob("*.nii.gz"))) # if re.search(json_re, p.as_posix()) != None)

    if len(json_paths) == 0 or len(nii_paths) == 0:
        exit_program_early("Could not find Nifti or JSON sidecar files in the bids directory.")

    if len(json_paths) != len(nii_paths):
        exit_program_early("Unequal amount of NIFTI and JSON files found")
 
    for p_json, p_nii in zip(json_paths, nii_paths):
        j = json.load(p_json.open()) 
        if quality_pairs[j["SeriesNumber"]] == "unusable":
            os.remove(p_json.as_posix()) 
            os.remove(p_nii.as_posix()) 


def run_dcm2bids(source_dir:Path, bids_output_dir:Path, config_file:Path, subject:str, session:str, config_file2:Path=None, nordic=False, nifti=False):
    if not source_dir.exists():
        exit_program_early(f"Path {source_dir} does not exist.")
    elif not bids_output_dir.exists():
        exit_program_early(f"Path {bids_output_dir} does not exist.")
    elif not config_file.exists():
        exit_program_early(f"Path {config_file} does not exist.")
    elif shutil.which('dcm2bids') == None:
        exit_program_early("Cannot locate program 'dcm2bids', make sure it is in your PATH.")

    helper_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                 --bids_validate 
                                 {'--skip_dcm2niix' if args.nifti else ''}
                                 -d {source_dir.as_posix()} 
                                 -p {subject} 
                                 -s {session} 
                                 -c {config_file.as_posix()} 
                                 -o {bids_output_dir.as_posix()}""")
    try:
        print("####### Running first round of Dcm2Bids ########")
        # subprocess.check_output(helper_command) # run helper command to generate json/.nii files; throw error if fail
        with subprocess.Popen(helper_command, stdout=subprocess.PIPE) as p:
            while p.poll() == None:
                text = p.stdout.read1().decode("utf-8", "ignore")
                print(text, end="", flush=True)
            if p.poll() != 0:
                raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
            
        if nordic and config_file2:
            if not config_file2.exists():
                exit_program_early(f"Path {config_file2} does not exist.")

            nordic_run_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                            --bids_validate
                                            {'--skip_dcm2niix' if nifti else ''}
                                            -d {source_dir.as_posix()} 
                                            -p {subject}
                                            -s {session}
                                            -c {config_file2.as_posix()}
                                            -o {bids_output_dir.as_posix()}
                                            """)
            print("####### Running second round of Dcm2Bids ########")
            # subprocess.check_output(nordic_run_command)
            with subprocess.Popen(nordic_run_command, stdout=subprocess.PIPE) as p:
                while p.poll() == None:
                    text = p.stdout.read1().decode("utf-8", "ignore")
                    print(text, end="", flush=True)
                if p.poll() != 0:
                    raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
                
    except RuntimeError or subprocess.CalledProcessError as e:
        print(e)
        exit_program_early("Problem running 'dcm2bids'.")
    

def dicom_to_bids(subject:str, session:str, source_dir:str, bids_dir:str, xml_path:str, bids_config1:str, bids_config2:str=None, nordic=False):
    source_dir = Path(source_dir)
    bids_dir = Path(bids_dir)
    xml_path = Path(xml_path)
    bids_config1 = Path(bids_config1)
    if nordic: 
        if bids_config2:
            bids_config2 = Path(bids_config2)
        if not bids_config2:
            print("---[WARNING]: Nordic flag set, but a second dcm2bids config file was not provided")
            prompt_user_continue("Continue without NORDIC processing?")
    if bids_config2:
        if not nordic:
            print("---[WARNING]: Second dcm2bids config file was provided, but the nordic flag was not set")
            prompt_user_continue("Continue without NORDIC processing?")

    run_dcm2bids(source_dir, bids_dir, bids_config1, subject, session, bids_config2, nordic)
    remove_unusable_runs(xml_path, bids_dir, subject)


if __name__ == "__main__":
    parser = ArgumentParser(prog="bids_wrapper.py",
                                    description="wrapper script for dcm2bids",
                                    epilog="WIP")
    parser.add_argument("-su", "--subject", required=True, help="Subject ID")
    parser.add_argument("-se","--session", required=True, help="Session ID")
    parser.add_argument("-sd", "--source_data", required=True, help="Path to directory containing this session's DICOM files")
    parser.add_argument("-b", "--bids_path", required=True, help="Path to the bids directory to store the newly made NIFTI files")
    parser.add_argument("-x", "--xml_path", required=True, help="Path to this session's XML file")
    parser.add_argument("-c1", "--bids_config1", required=True, help="dcm2bids config json file")
    parser.add_argument("-c2", "--bids_config2", help="Second dcm2bids config json file used for NORDIC processing")
    parser.add_argument("--nordic", action="store_true", help="Flag to indicate there are nordic runs in this data")
    # parser.add_argument("--nifti", help="specify that our DICOM folder contains .nii/.jsons instead", action='store_true')
    args = parser.parse_args()
    
    dicom_to_bids(subject=args.subject,
                  session=args.session,
                  source_dir=args.source_data,
                  bids_dir=args.bids_path,
                  xml_path=args.xml_path,
                  bids_config1=args.bids_config1,
                  bids_config2=args.bids_config2,
                  nordic=args.nordic)