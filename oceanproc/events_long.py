import pandas as pd
import numpy as np
import nibabel as nib
import json
from .utils import exit_program_early, debug_logging, log_linebreak
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
def make_events_long(event_file:Path, volumes: int, tr:float, output_file:Path = None):
    """
    Takes and event file and a funtional run and creates a long formatted events file
    that maps the onset of task events to a frame of the functional run

    :param func_data: A numpy array-like object representing functional data
    :type bold_run: npt.ArrayLike
    :param event_file: path to the event timing file
    :type event_file: pathlib.Path
    :param tr: Repetition time of the function run in seconds
    :type tr: float
    :param output_file: file path (including name) to save the long formatted event file to
    :type output_file: pathlib.Path
    """

    duration = tr * volumes
    events_df = pd.read_csv(event_file, index_col=None, sep="," if str(event_file).endswith(".csv") else "\t")
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(0, columns=conditions, index=np.arange(0,duration,tr)[:volumes])

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e,'onset'])
        events_long.loc[i, events_df.loc[e,'trial_type']] = 1
        if events_df.loc[e,'duration'] > tr:
            offset = events_df.loc[e,'onset'] + events_df.loc[e,'duration']
            j = find_nearest(events_long.index, offset)
            events_long.loc[i:j, events_df.loc[e,'trial_type']] = 1

    if output_file and output_file.suffix == ".csv":
        logger.debug(f" saving events long to file: {output_file}")
        events_long.to_csv(output_file)

    return events_long


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
    conf_df = pd.read_csv(confounds_file, sep="," if str(confounds_file).endswith('.csv') else "\t")
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
        search_path = f"sub-{sub}_ses-{ses}*"
        task = etf.name.split('task-')[-1].split('_')[0]
        search_path = f"{search_path}task-{task}*"
        run = None
        if "run" in etf.name:
            run = etf.name.split('run-')[-1].split('_')[0].zfill(2)
            search_path = f"{search_path}run-{run}*"
        bold_files = list(derivs_func.glob(f"{search_path}_bold.*nii*"))
        if len(bold_files) < 1:
            logger.info(f"Could not find any bold files that matched this event timing file: {etf}")
            continue
        confounds_file = list(derivs_func.glob(f"{search_path}confounds_timeseries.tsv"))
        if len(confounds_file) < 1:
            logger.info(f"Could not find any confounds files that matched this event timing file: {etf}")
            continue
        confounds_file = confounds_file[0]
        bold_file = bold_files[0]

        nvols = None
        for bf in bold_files:
            bold_file = bf
            if bold_file.name.endswith("bold.dtseries.nii"):
                nvols = nib.load(bold_file).shape[0]
                break
            elif bold_file.name.endswith("bold.nii.gz") or bold_file.name.endswith("bold.nii"):
                nvols = nib.load(bold_file).shape[-1]
                break
        if not nvols:
            logger.warning(f"Could not find any bold files of type '.dtseries.nii' or '.nii' or '.nii.gz' matching this event timing file: {etf}")
            continue

        side_car = bold_file.with_suffix("").with_suffix(".json")
        tr = None
        with open(side_car, "r") as f:
            jd = json.load(f)
            tr = jd["RepetitionTime"]

        logger.info(f"creating event file with file set: \n\t{etf}\n\t{bold_file}\n\t{confounds_file}")
            
        event_file_out = derivs_func / f"sub-{sub}_ses-{ses}_task-{task}_{f'run-{run}' if run else ''}_desc-events_long.csv"
        make_events_long(event_file=etf, 
                         volumes=nvols,
                         tr=tr,
                         output_file=event_file_out)
        append_to_confounds(confounds_file=confounds_file, 
                            fd_thresh=fd_thresh)
        




