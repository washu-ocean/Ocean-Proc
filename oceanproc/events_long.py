import pandas as pd
import numpy as np
import nibabel as nib
import json
import os
from .utils import exit_program_early, debug_logging, log_linebreak
from glob import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def find_nearest(array, value):
    """
    Finds the smallest difference in 'value' and one of the 
    elements of 'array', and returns the index of the element

    :param array: a list of elements to compare value to
    :type array: a list or list-like object
    :param value: a value to compare to elements of array
    :type value: integer or float
    :return: integer index of array
    :rtype: int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(array[idx])


@debug_logging
def make_events_long(bold_run:Path, event_file:Path, output_file:Path, tr:float):
    """
    Takes and event file and a funtional run and creates a long formatted events file
    that maps the onset of task events to a frame of the functional run

    :param bold_run: path to the function run corresponding to the event file
    :type bold_run: pathlib.Path
    :param event_file: path to the event timing file
    :type event_file: pathlib.Path
    :param output_file: file path (including name) to save the long formatted event file to
    :type output_file: pathlib.Path
    :param tr: Repetition time of the function run in seconds
    :type tr: float
    """
    nvols = nib.load(bold_run).dataobj.shape[-1]
    duration = nvols * tr

    events_df = pd.read_csv(event_file, index_col=None, delimiter="\t")
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(0, columns=conditions, index=np.arange(0,duration,tr))

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e, "onset"])
        events_long.loc[i, events_df.loc[e, "trial_type"]] = 1
        if events_df.loc[e, "duration"] > tr:
            offset = events_df.loc[e, "onset"] + events_df.loc[e, "duration"]
            j = find_nearest(events_long.index, offset)
            if j>i:
                events_long.loc[j, events_df.loc[e, "trial_type"]] = 1

    events_long.to_csv(output_file)


@debug_logging
def append_to_confounds(confounds_file:Path, fd_thresh:float):
    """
    Makes motion outlier regressors based on the framewise
    displacement threshold, and appends them to the confounds file
    for this functional run

    :param confounds_file: path to the confounds file for this functional run
    :type confounds_file: pathlib.Path
    :param fd_thresh: Framewise displacement threshold for this functional run
    :type fd_thresh: float
    """
    conf_df = pd.read_csv(confounds_file, delimiter="\t")
    b = 0
    for a in range(len(conf_df)):
        if conf_df.loc[a, "framewise_displacement"] > fd_thresh:
            conf_df[f"spike{b}"] = 0
            conf_df.loc[a, f"spike{b}"] = 1
            b += 1
    
    conf_df.to_csv(confounds_file, sep="\t")
    

@debug_logging
def create_events_and_confounds(bids_path:Path, derivs_path:Path, sub:str, ses:str, fd_thresh:float):
    """
    Facilitates the creation of a long formatted events file
    and the appending of motion outlier regressors to the 
    confounds file for each functional run for this subject and session

    :param bids_path: Path to the bids directory containing the raw
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param subject: Subject id (
    :type subject: str
    :param session: Session id 
    :type session: str
    :param fd_thresh: Framewise displacement threshold for this functional run
    :type fd_thresh: float
    
    """
    log_linebreak()
    logger.info("####### Creating long formatted event files ########\n")

    bids_func = bids_path / f"sub-{sub}/ses-{ses}/func"
    derivs_func = derivs_path / f"sub-{sub}/ses-{ses}/func"
    if not bids_func.is_dir():
        exit_program_early(f"Cannnot find 'func' - {bids_func} - bids directory for this subject and session")
    if not derivs_func.is_dir():
        exit_program_early(f"Cannnot find 'func' - {derivs_func} - derivatives directory for this subject and session")

    event_time_files = list(bids_func.glob("*_events.tsv"))
    logger.info(f"Found {len(event_time_files)} event timing files")
    for etf in event_time_files:
        search_path = f"/sub-{sub}_ses-{ses}*"
        task = etf.name.split('task-')[-1].split('_')[0]
        search_path = f"{search_path}task-{task}*"
        run = None
        if "run" in etf.name:
            run = etf.name.split('run-')[-1].split('_')[0].zfill(2)
            search_path = f"{search_path}run-{run}*"
        bold_file = list(bids_func.glob(f"{search_path}bold.nii*"))
        if len(bold_file) < 1:
            logger.info(f"Could not find any bold files that matched this event timing file: {etf}")
            continue
        confounds_file = list(derivs_func.glob(f"{search_path}confounds_timeseries.tsv"))
        if len(confounds_file) < 1:
            logger.info(f"Could not find any confounds files that matched this event timing file: {etf}")
            continue
        confounds_file = confounds_file[0]
        bold_file = bold_file[0]
        side_car = bold_file.split(".")[0] + ".json"
        tr = None
        with open(side_car, "r") as f:
            jd = json.load(f)
            tr = jd["RepetitionTime"]

        event_file_out = derivs_func / f"sub-{sub}_ses-{ses}_task-{task}_{f'run-{run}' if run else ''}_desc-events_long.csv"
        make_events_long(bold_run=bold_file, 
                         event_file=etf, 
                         output_file=event_file_out, 
                         tr=tr)
        append_to_confounds(confounds_file=confounds_file, 
                            fd_thresh=fd_thresh)
        




