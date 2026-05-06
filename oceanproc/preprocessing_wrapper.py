#!/usr/bin/env python3

from pathlib import Path
import logging
from bids import BIDSLayout
from .utils import exit_program_early, make_option, prepare_subprocess_logging, flags, debug_logging, log_linebreak, run_subprocess, update_permissions
import shutil
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as bsoup
import pandas as pd
import numpy as np
import nibabel as nib
import json
import os
import copy
from types import SimpleNamespace

logger = logging.getLogger(__name__)
plt.set_loglevel("warning")

adult_defaults = SimpleNamespace(
    image_name="nipreps/fmriprep",
    image_version="23.1.4",
    derivs_subfolder="fmriprep",
)

infant_defaults = SimpleNamespace(
    image_name="nipreps/nibabies",
    image_version="25.0.1",
    derivs_subfolder="nibabies",
)

# Preprocessing arguments that need to have their paths binded when running docker
mount_opts = {"fs_subjects_dir",
                  "derivatives"}

@debug_logging
def run_preprocessing(subject:str,
                      bids_path:Path,
                      derivs_path:Path,
                      work_path:Path,
                      license_file:Path,
                      image_name:str,
                      title:str,
                      option_chain:str,
                      additional_mounts:dict[str, ],
                      session:str = None,
                      remove_work_folder:bool = True):
    """
    Run preprocessing with parameters.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param work_path: Path to the working directory that will be deleted upon completion or error
    :type work_path: Path
    :param license_file: Path to the freesurfer license file
    :type license_file: Path
    :param image_name: Name of the docker image.
    :type image_name: str
    :param option_chain: String containing generated list of options built by make_option().
    :type option_chain: str
    :param remove_work_folder: Flag to indicate if the work_path should be deleted (default True)
    :type remove_work_folder: bool
    :raise RuntimeError: If preprocessing throws an error, or exits with a non-zero exit code.
    """
    def clean_up(): return shutil.rmtree(work_path) if remove_work_folder else None
   
    log_linebreak()
    logger.info(f"####### Starting {title} #######\n")
    if not bids_path.exists():
        exit_program_early(f"Bids path {bids_path} does not exist.")
    elif not derivs_path.exists():
        exit_program_early(f"Derivatives path {derivs_path} does not exist.")

    license_mount = "/opt/freesurfer/license.txt"
    bids_mount = "/data"
    derivs_mount = "/out"
    work_mount = "/work"
    
    if flags.apptainer:
        cmd_prelude = f"apptainer run --nv --cleanenv --no-mount cwd --pwd {work_mount}"
        mount_flag = "-B"
    else: 
        cmd_prelude = f"docker run --rm -i -u {flags.uid}:{flags.gid}"
        mount_flag = "-v"

    additional_mount_paths = []
    additional_mount_args = []
    mount_dex = 1
    for k, v in additional_mounts.items():
        arg_vals = [k]
        if isinstance(v, list):
            for sub_v in v:
                additional_mount_paths.append(f"{mount_flag} {sub_v.resolve()}:/deriv{mount_dex}")
                arg_vals.append(f"/deriv{mount_dex}")
                mount_dex += 1
        else:
            additional_mount_paths.append(f"{mount_flag} {v.resolve()}:/deriv{mount_dex}")
            arg_vals.append(f"/deriv{mount_dex}")
            mount_dex += 1
        additional_mount_args.append(" ".join(arg_vals))
    additional_mount_paths = " ".join(additional_mount_paths)
    additional_mount_args = " ".join(additional_mount_args)

    preproc_command = f"""{cmd_prelude}
                            {mount_flag} {license_file.resolve()}:{license_mount}:ro
                            {mount_flag} {bids_path.resolve()}:{bids_mount}:ro
                            {mount_flag} {derivs_path.resolve()}:{derivs_mount}
                            {mount_flag} {work_path.resolve()}:{work_mount}
                            {additional_mount_paths}
                            {image_name} {bids_mount} {derivs_mount} participant
                            --participant-label {subject} 
                            {f'--session-id {session}' if session else ''}
                            -w {work_mount} --verbose
                            {additional_mount_args}
                            {option_chain}"""
    
    success = True
    try:
        run_subprocess(preproc_command, title=title)
    except (Exception, KeyboardInterrupt) as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        success = False

    if not flags.debug:
        clean_up()

    return success

@debug_logging
def add_fd_plot_to_report(subject:str,
                          session:str,
                          derivs_path:Path,
                          title:str):
    """
    Reads each confounds file in the preprocessing functional output, plots the framewise
    displacement, and adds this figure into the output report

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    """

    func_path = derivs_path / f"sub-{subject}" / f"ses-{session}" / "func"
    figures_path = derivs_path / f"sub-{subject}" / "figures"
    report_path = derivs_path / f"sub-{subject}{f'_ses-{session}' if flags.infant else ''}.html"

    for p in [func_path, figures_path, report_path]:
        if not p.exists():
            exit_program_early(f"Expected Path {str(p)} does not exist.")

    log_linebreak()
    logger.info(f"####### Appending FD Plots to {title} Report #######\n")

    logger.debug(f"parsing the {title} html report file: {report_path}")
    report_file = open(report_path, "r")
    soup = bsoup(report_file.read(), 'html.parser')
    report_file.close()

    for f in func_path.glob("*desc-confounds_timeseries.tsv"):
        logger.info(f"plotting framewise displacement from confounds file:{f}")
        try:
            run = f.name.split("run-")[-1].split("_")[0]
            html_run = int(run) if flags.infant else run
            task = f.name.split("task-")[-1].split("_")[0]

            # read the repetition time from the json files for the BOLD data
            bold_js = list(f.parent.glob(f"*task-{task}*run-{run}*bold.json"))[0]
            logger.debug(f" reading repetition time from JSON file: {bold_js}")
            tr = None
            with open(bold_js, "r") as jf:
                tr = float(json.load(jf)["RepetitionTime"])

            # read in the confounds file
            confound_df = pd.read_csv(f, sep="\t")
            n_frames = len(confound_df["framewise_displacement"])
            x_vals = np.arange(0, n_frames * tr, tr)
            mean_fd = np.mean(confound_df["framewise_displacement"])
            func_thresh = 0.9
            rest_thresh = 0.2

            # plot the framewise displacement
            fig, ax = plt.subplots(1,1, figsize=(15,5))
            # ax.set_ylabel("Displacement (mm)")
            ax.set_xlabel("Time (sec)")
            ax.plot(x_vals, confound_df["framewise_displacement"], label="FD Trace")
            ax.plot(x_vals, [func_thresh] * n_frames, label=f"Threshold: {func_thresh}")
            ax.plot(x_vals, [rest_thresh] * n_frames, label=f"Threshold: {rest_thresh}")
            ax.plot(x_vals, [mean_fd] * n_frames, label=f"Mean: {round(mean_fd,2)}")
            ax.set_xlim(0, (n_frames * tr))
            ax.set_ylim(0, 1.5)
            ax.legend(loc="upper left")
            plot_path = figures_path / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-fd-trace.svg"
            fig.savefig(plot_path, bbox_inches="tight", format="svg", pad_inches=0.2)
            logger.debug(f" saved the fd plot figure for run-{run} to path: {plot_path}")

            # find the location in the report where the new figure will go
            confounds_plot_div = soup.find(id=lambda x: ((f"desc-carpetplot" in x) and (f"run-{html_run}" in x)) if x else False)

            # Copy a div element from the report and add the new figure into it
            fd_plot_div = copy.copy(confounds_plot_div)
            del fd_plot_div["id"]
            fd_plot_div.p.extract()
            fd_plot_div.h3.string = "Scaled FD Plot"
            rel_path = os.path.relpath(plot_path, derivs_path)
            fd_plot_div.img["src"] = "./" + rel_path

            # find the reference div for the copied div element and make a copy of this as well
            if flags.infant:
                fd_plot_reference_div = fd_plot_div.small
                fd_plot_reference_div.a["href"] = "./" + rel_path
                fd_plot_reference_div.a.string = rel_path

                # Add the new elements into the file
                logger.debug(f" inserting the new html elements into the {title} report")
                confounds_plot_div.insert_after(fd_plot_div)
            else: 
                confounds_plot_reference_div = confounds_plot_div.find_next_sibling("div", class_="elem-filename")
                fd_plot_reference_div = copy.copy(confounds_plot_reference_div)
                fd_plot_reference_div.a["href"] = "./" + rel_path
                fd_plot_reference_div.a.string = rel_path

                # Add the new elements into the file
                logger.debug(f" inserting the new html elements into the {title} report")
                confounds_plot_reference_div.insert_after(fd_plot_div)
                fd_plot_div.insert_after(fd_plot_reference_div)
        except Exception as e:
            logger.warning(f"Error generating the scaled FD plot for confound file: {f}")

    logger.info("writing the edited html to the report file")
    with open(report_path, "w") as f:
        f.write(soup.prettify())


def insert_dummy_frames(subject:str,
                        session:str,
                        dscan_dir:Path,
                        bids_layout:BIDSLayout):
    
    dscans_suffix = "dscans"
    dscans_path_pattern = "sub-{subject}[/ses-{session}]/{datatype<func|meg|beh>|func}/sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}][_recording-{recording}]_{suffix<"+dscans_suffix+">}{extension<.tsv|.json>|.tsv}"

    log_linebreak()
    logger.info(f"####### Inserting Dummy Scans #######\n")

    # find all bold files
    func_files = bids_layout.get(subject=subject, session=session, suffix="bold", datatype="func", extension="nii.gz")
    for bfile in func_files:
        if not Path(bfile.path).exists:
            continue

        # Find the dscans file for bold run
        try_find = True
        dscans_entities = bids_layout.parse_file_entities(bfile.path) | {"suffix":dscans_suffix, "extension":".tsv"}
        while try_find:
            dscans_bids_path = bids_layout.build_path(dscans_entities, path_patterns=[dscans_path_pattern], validate=False)
            dscans_file = sorted(dscan_dir.glob(Path(dscans_bids_path).name))
            if len(dscans_file) > 1:
                exit_program_early(f"Found more than one dscans file for bold run {bfile} --> {dscans_file}")
            elif len(dscans_file) == 1:
                try_find = False
            if ("echo" in dscans_entities) and try_find:
                del dscans_entities["echo"]
            else:
                try_find = False

        if len(dscans_file) == 0:
            logger.info(f"Did not find any dscans files for bold run: {bfile}, file will remain unmodified.")
            continue
        else:
            dscans_file = dscans_file[0]
        logger.info(f"found dscans file <{dscans_file.name}> for bold run <{bfile.filename}>")

        dscans_list = pd.read_csv(dscans_file, sep="\t").loc[:, "dummy_scan"].to_numpy().astype(np.int32)
        bold_img = nib.load(bfile.path)
        num_bold_frames = bold_img.shape[3]
        num_dummy_scans = np.sum(dscans_list > 0)
        num_non_dummy = np.sum(dscans_list == 0)

        # validate length
        if len(dscans_list) == num_bold_frames:
            logger.info(f"  dsans file and bold image have the same length. Dummy scans may have already been inserted or length of dscans file is incorrect. Bold file will remain unmodified.")
            continue
        elif num_bold_frames != num_non_dummy:
            exit_program_early(f"Number of non-dummy scans and length of bold data do not match. # non-dummy scans: {num_non_dummy}, # bold frames: {num_bold_frames}")

        logger.info(f"  inserting {num_dummy_scans} dummy scans into the bold run")
        # create a new bold series with dummy scans inserted
        bold_data = bold_img.get_fdata()
        frame_list = []
        dscans_index = 0
        for f in range(num_bold_frames):
            copy_dex = f-1 if f > 0 else f
            while dscans_list[dscans_index] != 0:
                frame_list.append(bold_data[:,:,:,copy_dex])
                dscans_index += 1
            frame_list.append(bold_data[:,:,:,f])
            dscans_index += 1
            # check if end of run needs to be extended
            if (f == num_bold_frames-1) and (dscans_index < len(dscans_list)):
                while dscans_index < len(dscans_list):
                    frame_list.append(bold_data[:,:,:,f])
                    dscans_index += 1

        if len(frame_list) != len(dscans_list):
            exit_program_early(f"Something went wrong when creating dummy scans, lengths do not match. bold frames: {len(frame_list)}, dscan length: {len(dscans_list)}")
        
        # save new image
        extended_bold_data = np.stack(frame_list, axis=3)
        extended_bold_img = bold_img.__class__(extended_bold_data, header=bold_img.header, affine=bold_img.affine)
        logger.info(f"  saving extended bold image")
        nib.save(extended_bold_img, bfile.path)
    

@debug_logging
def process_data(subject:str,
                 session:str,
                 bids_path:Path,
                 derivs_path:Path,
                 work_path:Path,
                 license_file:Path,
                 image_name:str,
                 additional_mounts:dict[str, Path],
                 remove_work_folder:bool,
                 **kwargs):
    """
    Faciliates the running of preprocessing and any additions to the output report.

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
    :param image_name: Name of the docker image.
    :type image_name: str
    :param license_file: Path to the freesurfer license file
    :type license_file: Path
    :param remove_work_folder: Path to the working directory that will be deleted upon completion or error (default None)
    :type remove_work_folder: str
    :param **kwargs: any arguments to be passed to the preprocessing subprocess
    """
    option_chain = " ".join([make_option(v, key=k, delimeter=" ", convert_underscore=True) for k,v in kwargs.items()])
    preproc_title = "NiBabies" if flags.infant else "fMRIPrep"

    preproc_success = run_preprocessing(subject=subject,
                      session=session if flags.infant else None,
                      bids_path=bids_path,
                      derivs_path=derivs_path,
                      work_path=work_path,
                      license_file=license_file,
                      image_name=image_name,
                      title=preproc_title,
                      option_chain=option_chain,
                      additional_mounts=additional_mounts,
                      remove_work_folder=remove_work_folder)
    
    if preproc_success:
        add_fd_plot_to_report(subject=subject,
                            session=session,
                            derivs_path=derivs_path,
                            title=preproc_title)
    
    paths_to_update = []
    paths_to_update.extend(sorted(derivs_path.glob(f"sub-{subject}*")))
    paths_to_update.extend(sorted(derivs_path.glob(f"sourcedata/*/sub-{subject}*")))
    for out_path in paths_to_update:
        if out_path.exists():
            # update file permissions for preprocessing outputs
            update_permissions(
                permissions=flags.file_permissions, 
                path=out_path, 
                recursive=True,
                group=flags.permissions_group
            )
        
    # check other top-level files
    other_paths = [derivs_path/"dataset_description.json", derivs_path/"logs"]
    for op in other_paths:
        if op.exists() and (op.stat().st_uid == flags.uid):
            update_permissions(
                permissions=flags.file_permissions,
                path=op,
                recursive=True,
                group=flags.permissions_group
            )
            
    # update file permisions for working directory files
    update_permissions(
        permissions=flags.file_permissions, 
        path=work_path, 
        recursive=True,
        group=flags.permissions_group,
        quiet=True
    )
    
    if not preproc_success:
        exit_program_early(f"Program '{preproc_title}' has run into an error.")