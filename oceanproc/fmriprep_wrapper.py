#!/usr/bin/env python3

from pathlib import Path
import logging
from .utils import exit_program_early, make_option, prepare_subprocess_logging, flags, debug_logging, log_linebreak
import shlex
import shutil
from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as bsoup
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
import copy

logger = logging.getLogger(__name__)
plt.set_loglevel("warning")


@debug_logging
def run_fmri_prep(subject:str,
                  bids_path:Path,
                  derivs_path:Path,
                  option_chain:str,
                  remove_work_folder:Path = None,
                  unknown_args: list = None):
    """
    Run fmriprep with parameters.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param option_chain: String containing generated list of options built by make_option().
    :type option_chain: str
    :param remove_work_folder: Path to the working directory that will be deleted upon completion or error (default None)
    :type remove_work_folder: str
    :raise RuntimeError: If fmriprep throws an error, or exits with a non-zero exit code.
    """
    def clean_up(): return shutil.rmtree(remove_work_folder) if remove_work_folder else None

    log_linebreak()
    logger.info("####### Starting fMRIPrep #######\n")
    if not bids_path.exists():
        exit_program_early(f"Bids path {bids_path} does not exist.")
    elif not derivs_path.exists():
        exit_program_early(f"Derivatives path {derivs_path} does not exist.")
    elif shutil.which('fmriprep-docker') == None:
        exit_program_early("Cannot locate program 'fmriprep-docker', make sure it is in your PATH.")

    uid = Popen(["id", "-u"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    gid = Popen(["id", "-g"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    cifti_out_res = '91k'

    helper_command = shlex.split(f"""{shutil.which('fmriprep-docker')} 
                                 --user {uid}:{gid}
                                 --participant-label={subject}
                                 --cifti-output={cifti_out_res}
                                 --use-syn-sdc=warn
                                 {option_chain}
                                 {str(bids_path)}
                                 {str(derivs_path)}
                                 participant
                                 {'' if unknown_args is None else ' '.join(unknown_args)}
                                 """)
    try:
        command_str = " ".join(helper_command)
        logger.info(f"Running fmriprep-docker with the following command: \n  {command_str} \n")
        prepare_subprocess_logging(logger)
        with Popen(helper_command, stdout=PIPE) as p:
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'fmriprep-docker' has ended with a non-zero exit code.")
    except RuntimeError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early("Program 'fmriprep-docker' has run into an error.",
                           None if flags.debug else clean_up)
    if not flags.debug:
        clean_up()


@debug_logging
def add_fd_plot_to_report(subject:str,
                          session:str,
                          derivs_path:Path):
    """
    Reads each confounds file in the fmriprep functional output, plots the framewise
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
    report_path = derivs_path / f"sub-{subject}.html"

    for p in [func_path, figures_path, report_path]:
        if not p.exists():
            exit_program_early(f"Expected Path {str(p)} does not exist.")

    log_linebreak()
    logger.info("####### Appending FD Plots to fMRIPrep Report #######\n")

    logger.debug(f"parsing the fMRIPrep html report file: {report_path}")
    report_file = open(report_path, "r")
    soup = bsoup(report_file.read(), 'html.parser')
    report_file.close()

    for f in func_path.glob("*desc-confounds_timeseries.tsv"):
        logger.info(f"plotting framewise displacement from confounds file:{f}")
        try:
            run = f.name.split("run-")[-1].split("_")[0]
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
            confounds_plot_div = soup.find(id=lambda x: (f"desc-carpetplot_run-{run}" in x) if x else False)

            # Copy a div element from the report and add the new figure into it
            fd_plot_div = copy.copy(confounds_plot_div)
            del fd_plot_div["id"]
            fd_plot_div.p.extract()
            fd_plot_div.h3.string = "Scaled FD Plot"
            rel_path = os.path.relpath(plot_path, derivs_path)
            fd_plot_div.img["src"] = "./" + rel_path

            # find the reference div for the copied div element and make a copy of this as well
            confounds_plot_reference_div = confounds_plot_div.find_next_sibling("div", class_="elem-filename")

            fd_plot_reference_div = copy.copy(confounds_plot_reference_div)
            fd_plot_reference_div.a["href"] = "./" + rel_path
            fd_plot_reference_div.a.string = rel_path

            # Add the new elements into the file
            logger.debug(f" inserting the new html elements into the fMRIPrep report")
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
                 remove_work_folder:Path = None,
                 unknown_args:list = None,
                 **kwargs):
    """
    Faciliates the running of fmriprep and any additions to the output report.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param remove_work_folder: Path to the working directory that will be deleted upon completion or error (default None)
    :type remove_work_folder: str
    :param **kwargs: any arguments to be passed to the fmriprep subprocess
    """

    fmriprep_option_chain = " ".join([make_option(v, key=k, delimeter="=", convert_underscore=True) for k,v in kwargs.items()])

    run_fmri_prep(subject=subject,
                  bids_path=bids_path,
                  derivs_path=derivs_path,
                  option_chain=fmriprep_option_chain,
                  remove_work_folder=remove_work_folder,
                  unknown_args=unknown_args)

    add_fd_plot_to_report(subject=subject,
                          session=session,
                          derivs_path=derivs_path)
