#!/usr/bin/env python3

from pathlib import Path
import logging
from .utils import exit_program_early, make_option, prepare_subprocess_logging, flags, debug_logging, log_linebreak, run_subprocess
import shutil
from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as bsoup
import pandas as pd
import numpy as np
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

# Good arguments to have if not specified
# { key : (val, can_have_multiple_values)}
preproc_kwargs = {
    "cifti-output":("91k", False),
    "output-spaces": ("fsLR", True),
    "output-spaces": ("MNI152NLin2009cAsym", True)
}

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
    
    additional_mount_paths = []
    additional_mount_args = []
    mount_dex = 1
    for k, v in additional_mounts.items():
        arg_vals = [k]
        if isinstance(v, list):
            for sub_v in v:
                additional_mount_paths.append(f"-v {sub_v.resolve()}:/deriv{mount_dex}")
                arg_vals.append(f"/deriv{mount_dex}")
                mount_dex += 1
        else:
            additional_mount_paths.append(f"-v {v.resolve()}:/deriv{mount_dex}")
            arg_vals.append(f"/deriv{mount_dex}")
            mount_dex += 1
        additional_mount_args.append(" ".join(arg_vals))
    additional_mount_paths = " ".join(additional_mount_paths)
    additional_mount_args = " ".join(additional_mount_args)

    uid = Popen(["id", "-u"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    gid = Popen(["id", "-g"], stdout=PIPE).stdout.read().decode("utf-8").strip()

    license_mount = "/opt/freesurfer/license.txt"
    bids_mount = "/data"
    derivs_mount = "/out"
    work_mount = "/work"
    preproc_command = f"""docker run --rm -i -u {uid}:{gid}
                            -v {license_file.resolve()}:{license_mount}:ro
                            -v {bids_path.resolve()}:{bids_mount}:ro
                            -v {derivs_path.resolve()}:{derivs_mount}
                            -v {work_path.resolve()}:{work_mount}
                            {additional_mount_paths}
                            {image_name} {bids_mount} {derivs_mount} participant
                            --participant-label {subject} 
                            {f'--session-id {session}' if session else ''}
                            -w {work_mount} --verbose
                            {additional_mount_args}
                            {option_chain}"""
    
    try:
        run_subprocess(preproc_command, title=title)
    except RuntimeError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early(f"Program '{title}' has run into an error.",
                           None if flags.debug else clean_up)
    if not flags.debug:
        clean_up()


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

    # Add in some recommended options
    for key, (val, multi) in preproc_kwargs.items():
        if (key in kwargs) and multi:
            if isinstance(kwargs[key], list):
                if val not in kwargs[key]:
                    kwargs[key].append(val)
            else:
                if val != kwargs[key]:
                    kwargs[key] = [kwargs[key], val]
        elif key not in kwargs:
            kwargs[key] = val


    option_chain = " ".join([make_option(v, key=k, delimeter=" ", convert_underscore=True) for k,v in kwargs.items()])
    preproc_title = "NiBabies" if flags.infant else "fMRIPrep"

    run_preprocessing(subject=subject,
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

    add_fd_plot_to_report(subject=subject,
                          session=session,
                          derivs_path=derivs_path,
                          title=preproc_title)
