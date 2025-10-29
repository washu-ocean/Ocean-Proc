#!/usr/bin/env python3
import sys
import os
import os.path as op
import re
from pathlib import Path
import json
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import numpy as np
import nilearn.masking as nmask
from nilearn.signal import butterworth, _handle_scrubbed_volumes
from scipy import signal
from scipy.stats import gamma
from ..oceanparse import OceanParser
from ..events_long import make_events_long
from ..utils import exit_program_early, add_file_handler, default_log_format, export_args_to_file, flags, debug_logging, log_linebreak, load_data, prompt_user_continue, parcellate_dtseries
import logging
import argparse
import datetime
from textwrap import dedent


"""
TODO:
    * Find good way to pass hrf peak and undershoot variables
    * Save final noise df for each run
    * Debug Mode - save intermediate outputs
    * Options for highpass and lowpass filters
    * Function documentation and testing

"""
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)],
                    format=default_log_format)
logger = logging.getLogger()

VERSION = "1.1.3"


@debug_logging
def create_image(data: npt.ArrayLike,
                 imagetype: str,
                 brain_mask: str = None,
                 tr: float = None,
                 header: nib.cifti2.cifti2.Cifti2Header = None):
    img = None
    suffix = ".nii"
    d32k = 32492
    if imagetype == "nifti":
        img = nmask.unmask(data, brain_mask)
    elif imagetype == "cifti":
        ax0 = None
        if data.shape[0] > 1 and tr:
            ax0 = nib.cifti2.cifti2_axes.SeriesAxis(
                start=0.0,
                step=tr,
                size=data.shape[0]
            )
            suffix = (".ptseries" if flags.parcellated else ".dtseries") + suffix
        elif data.shape[0] == 1:
            ax0 = nib.cifti2.cifti2_axes.ScalarAxis(
                name=["beta"]
            )
            suffix = (".pscalar" if flags.parcellated else ".dscalar") + suffix
        else:
            raise RuntimeError("TR not supplied or data shape is incorrect")
        ax1 = None
        if header:
            ax1 = header.get_axis(1)
        else:
            # Need to change this default behavior to create a correct Brain Model axis
            ax1 = nib.cifti2.cifti2_axes.BrainModelAxis(
                name=(['CIFTI_STRUCTURE_CORTEX_LEFT'] * d32k) + (['CIFTI_STRUCTURE_CORTEX_RIGHT'] * d32k),
                vertex=np.concatenate((np.arange(d32k), np.arange(d32k))),
                nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT':d32k, 'CIFTI_STRUCTURE_CORTEX_RIGHT':d32k}
            )
        img = nib.cifti2.cifti2.Cifti2Image(data, (ax0, ax1))
    return (img,suffix)


def demean_detrend(func_data: npt.ArrayLike) -> np.ndarray:
    """
    Subtracts the mean and a least-squares-fit line from each timepoint at every vertex/voxel.

    Parameters
    ----------

    func_data: npt.ArrayLike
        array containing functional timeseries data

    Returns
    -------
    data_dd: np.ndarray
        A demeaned/detrended copy of the input array
    """
    data_dd = signal.detrend(func_data, axis=0, type='linear')
    return data_dd


def percent_signal_change(data: npt.ArrayLike, mask: npt.ArrayLike):
    masked_data = data[mask, :]
    mean = np.nanmean(masked_data, axis=0)
    mean = np.repeat(mean[np.newaxis,:], data.shape[0], axis=0)
    psc_data = ((data - mean) / np.abs(mean)) * 100
    non_valid_indices = np.where(~np.isfinite(psc_data))
    if len(non_valid_indices[0]) > 0:
        logger.warning("Found vertices with zero signal, setting these to zero")
    psc_data[np.where(~np.isfinite(psc_data))] = 0

    return psc_data


def avg_tsnr(fdata, axis=0):
    tsnr = np.nanmean(fdata, axis=axis) / np.nanstd(fdata, axis=axis)
    return np.nanmean(tsnr)


def create_hrf(time, time_to_peak=5, undershoot_dur=12):
    """
    This function creates a hemodynamic response function timeseries.

    Parameters
    ----------
    time: numpy array
        a 1D numpy array that makes up the x-axis (time) of our HRF in seconds
    time_to_peak: int
        Time to HRF peak in seconds. Default is 5 seconds.
    undershoot_dur: int
        Duration of the post-peak undershoot. Default is 12 seconds.

    Returns
    -------
    hrf_timeseries: numpy array
        The y-values for the HRF at each time point
    """

    peak = gamma.pdf(time, time_to_peak)
    undershoot = gamma.pdf(time, undershoot_dur)
    hrf_timeseries = peak - 0.35 * undershoot
    return hrf_timeseries


@debug_logging
def hrf_convolve_features(features: pd.DataFrame,
                          column_names: list = None,
                          time_col: str = 'index',
                          units: str = 's',
                          time_to_peak: int = 5,
                          undershoot_dur: int = 12,
                          custom_hrf: Path = None):
    """
    This function convolves a hemodynamic response function with each column in a timeseries dataframe.

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use; if it is None, use all columns. Default is None.
    time_col: str
        The name of the time column to use if not the index. Default is "index".
    units: str
        Must be 'ms','s','m', or 'h' to denote milliseconds, seconds, minutes, or hours respectively.
    time_to_peak: int
        Time to peak for HRF model. Default is 5 seconds.
    undershoot_dur: int
        Undershoot duration for HRF model. Default is 12 seconds.

    Returns
    -------
    convolved_features: DataFrame
        The HRF-convolved feature timeseries
    """
    if not column_names:
        column_names = features.columns

    if time_col == 'index':
        time = features.index.to_numpy()
    else:
        time = features[time_col]
        features.index = time

    if units == 'm' or units == 'minutes':
        features.index = features.index * 60
        time = features.index.to_numpy()
    if units == 'h' or units == 'hours':
        features.index = features.index * 3600
        time = features.index.to_numpy()
    if units == 'ms' or units == 'milliseconds':
        features.index = features.index / 1000
        time = features.index.to_numpy()

    convolved_features = pd.DataFrame(index=time)
    hrf_sig = np.loadtxt(custom_hrf) if custom_hrf is not None else create_hrf(time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)

    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features


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
    return (array[idx])


@debug_logging
def make_noise_ts(confounds_file: Path,
                  confounds_columns: list,
                  demean: bool = False,
                  linear_trend: bool = False,
                  spike_threshold: float = None,
                  volterra_expansion: int = None,
                  volterra_columns: list = None):
    select_columns = set(confounds_columns)
    fd = "framewise_displacement"
    if volterra_columns:
        select_columns.update(volterra_columns)
    if spike_threshold:
        select_columns.add(fd)
    nuisance = pd.read_csv(confounds_file, sep="," if str(confounds_file).endswith(".csv") else "\t").loc[:,list(select_columns)]
    if fd in select_columns:
        nuisance.loc[0, fd] = 0

    if demean:
        nuisance["mean"] = 1

    if linear_trend:
        nuisance["trend"] = np.arange(0, len(nuisance))

    """
    Add a new column denoting indices where a frame
    is censored for each row in the framewise_displacement
    that is larger than spike_threshold.
    """
    if spike_threshold:
        b = 0
        for a in range(len(nuisance)):
            if nuisance.loc[a,fd] > spike_threshold:
                nuisance[f"spike{b}"] = 0
                nuisance.loc[a, f"spike{b}"] = 1
                b += 1
        if fd not in confounds_columns:
            nuisance.drop(columns=fd, inplace=True)

    if volterra_expansion and volterra_columns:
        for vc in volterra_columns:
            for lag in range(volterra_expansion):
                nuisance.loc[:, f"{vc}_{lag + 1}"] = nuisance.loc[:, vc].shift(lag + 1)
        nuisance.fillna(0, inplace=True)
    elif volterra_expansion:
        raise RuntimeError("You must specify which columns you'd like to apply Volterra expansion to.")
    elif volterra_columns:
        raise RuntimeError("You must specify the lag applied in Volterra expansion.")

    return nuisance


# TODO: maybe write a validator for the input task file?
@debug_logging
def events_to_design(events_long: pd.DataFrame,
                     fir: int = None,
                     hrf: list[int] | Path = None,
                     fir_list: list[str] = None,
                     hrf_list: list[str] = None,
                     unmodeled_list: list[str] = None,
                     output_path: Path = None):
    """
    Builds an initial design matrix from an events long DataFrame. You can
    convolve specified event types with an hemodynamic response function
    (HRF).

    The events are expected to be in a .tsv file, with the following columns:

    trial_type: a string representing the unique trial type. oceanfla gives you the
        freedom to arrange these trial types in any way you want; they can represent
        events on their own, combinations of concurrent events, etc.
    onset: the onset time of an event
    duration: the duration of an event

    Parameters
    ----------
    events_long: pd.DataFrame
        A dataframe representing a long formatted events file
    fir: int = None
        An int denoting the order of an FIR filter
    hrf: tuple[int] = None
        A 2-length tuple, where hrf[0] denotes the time to the peak of an HRF, and hrf[1] denotes the duration of its "undershoot" after the peak.
    fir_list: list[str] = None
        A list of column names denoting which columns should have an FIR filter applied.
    hrf_list: list[str] = None
        A list of column names denoting which columns should be convolved with the HRF function defined in the hrf tuple.
    unmodeled_list: list[str] = None
        A list of column names denoting which columns should not be modeled by neither hrf or fir, but still included in the design matrix.
    output_path: pathlib.Path = None
        A path to a csv file where the created events matrix should be saved to.

    Returns
    -------
    (events_matrix, conditions): tuple
        A tuple containing the DataFrame with filtered/convolved columns and a list of unique trial names.
    """
    events_matrix = events_long.copy()
    if fir and fir <= 0:
        raise ValueError(f"fir value must be greater than 0. Current fir: {fir}")
    if hrf:
        if not (isinstance(hrf, Path) or ((len(hrf) == 2 and hrf[0] > 0 and hrf[1] > 0))):
            raise ValueError(f"hrf tuple must contain two integers greater than 0 or be the path to a txt file. Current hrf: {hrf}")
    # If both FIR and HRF are specified, we should have at least one list
    # of columns for one of the categories specified.
    if (fir and hrf) and not (fir_list or hrf_list):
        raise RuntimeError("Both FIR and HRF were specified, but you need to specify at least one list of variables (fir_list or hrf_list)")
    # fir_list and hrf_list must not have overlapping columns
    if (fir_list and hrf_list) and not set(fir_list).isdisjoint(hrf_list):
        raise RuntimeError("Both FIR and HRF lists of variables were specified, but they overlap.")
    conditions = [s for s in np.unique(events_matrix.columns)]  # unique trial types
    residual_conditions = [c for c in conditions if c not in unmodeled_list] if unmodeled_list else conditions
    if (fir and hrf) and (bool(fir_list) ^ bool(hrf_list)):  # Create other list if only one is specified
        if fir_list:
            hrf_list = [c for c in residual_conditions if c not in fir_list]
        elif hrf_list:
            fir_list = [c for c in residual_conditions if c not in hrf_list]
        assert set(hrf_list).isdisjoint(fir_list)

    if fir:
        fir_conditions = residual_conditions
        if fir_list and len(fir_list) > 0:
            fir_conditions = [c for c in residual_conditions if c in fir_list]
        residual_conditions = [c for c in residual_conditions if c not in fir_conditions]

        col_names = {c:c + "_00" for c in fir_conditions}
        events_matrix = events_matrix.rename(columns=col_names)
        fir_cols_to_add = dict()
        for c in fir_conditions:
            for i in range(1, fir):
                fir_cols_to_add[f"{c}_{i:02d}"] = np.array(np.roll(events_matrix.loc[:,col_names[c]], shift=i, axis=0))
                # so events do not roll back around to the beginnin
                fir_cols_to_add[f"{c}_{i:02d}"][:i] = 0
        events_matrix = pd.concat([events_matrix, pd.DataFrame(fir_cols_to_add, index=events_matrix.index)], axis=1)
        events_matrix = events_matrix.astype(int)
    if hrf:
        hrf_conditions = residual_conditions
        if hrf_list and len(hrf_list) > 0:
            hrf_conditions = [c for c in residual_conditions if c in hrf_list]
        residual_conditions = [c for c in residual_conditions if c not in hrf_conditions]

        cfeats = hrf_convolve_features(features=events_matrix,
                                       column_names=hrf_conditions,
                                       time_to_peak=hrf[0] if isinstance(hrf, list) else None,
                                       undershoot_dur=hrf[1] if isinstance(hrf, list) else None,
                                       custom_hrf=hrf if isinstance(hrf, Path) else None)
        for c in hrf_conditions:
            events_matrix[c] = cfeats[c]

    if len(residual_conditions) > 0 and logger:
        logger.warning(dedent(f"""The following trial types were not selected under either of the specified models
                           and were also not selected to be left unmodeled. These variables will not be included in the design matrix:\n\t {residual_conditions}"""))
        events_matrix = events_matrix.drop(columns=residual_conditions)

    if output_path:
        logger.debug(f" saving events matrix to file: {output_path}")
        events_matrix.to_csv(output_path)

    return (events_matrix, conditions)


@debug_logging
def filter_data(func_data: npt.ArrayLike,
                mask: npt.ArrayLike,
                tr: float,
                low_pass: float = 0.1,
                high_pass: float = 0.008,
                padtype: str = "zero",
                padlen: int = 50):
    """
    Apply a lowpass, highpass, or bandpass filter to the functional data. Masked frames
    are interploated using a cubic splice function before filtering. The returned array
    is the functional data after interpolation and filtering.

    Parameters
    ----------
    func_data: npt.ArrayLike

        A numpy array representing BOLD data

    mask: npt.ArrayLike

        A numpy array representing a mask along the first axis (time axis) of the BOLD data

    tr: float

        Repetition time at the scanner

    high_cut: float

        Frequency above which the bandpass filter will be applied

    low_cut: float

        Frequency below which the bandpass filter will be applied

    padtype: str or None

        Type of padding used in butterworth filter.
        Choices: "odd" (default), "even", "constant", "zero", or "none".

        "zero" padding is the same as "constant", just with zeroes appended to either side
        of the timeseries, since "constant" pads by the last element on either end of
        the timeseries.

    padlen: int or None

        Length of pad -- if None, default from `scipy.signal.filtfilt` will be used.

    Returns
    -------

    filtered_data: npt.ArrayLike

        A numpy array representing BOLD data with the filter applied
    """
    if not mask.shape[0] == func_data.shape[0]:
        raise ValueError("Mask must be the same length as the functional data")
    if not any((
        padtype == "none",
        padlen is None,
        padlen == -1,
        (padtype != "zero" and padlen is not None and padlen > 0),
        (padtype == "zero" and padlen is not None and padlen >= 2),
    )):
        raise ValueError(f"Pad length of {padlen} incompatible with pad type {'odd' if padtype is None else padtype}")

    # temporary fix for filter padding --- THIS MANUAL PADDING SEEMS TO FIX THE BIMODAL BETA DISTRIBUTION ---- ADDITIONAL TESTING MAY STILL BE NEEDED
    # padlen = 50
    # padtype = "zero"

    if padlen > 0 and padtype == "zero":
        padded_func_data = np.pad(func_data, ((padlen, padlen), (0, 0)), mode='constant', constant_values=0)
        padded_mask = np.pad(mask, (padlen, padlen), mode='constant', constant_values=True)
    else:
        padded_func_data = func_data
        padded_mask = mask

    if padlen == -1:
        padlen = None

    # if the mask is excluding frames, interpolate the censored frames
    if np.sum(mask) < mask.shape[0]:
        padded_func_data, _, padded_mask = _handle_scrubbed_volumes(
            signals=padded_func_data,
            confounds=None,
            sample_mask=padded_mask,
            filter_type="butterworth",
            t_r=tr,
            extrapolate=True
        )

    if padtype == "zero":
        # padded_func_data = np.pad(func_data, ((padlen, padlen), (0, 0)), mode='constant', constant_values=0)
        # if padlen is not None:
        #     padlen -= 2
        filtered_data = butterworth(
            signals=padded_func_data,
            sampling_rate=1.0 / tr,
            low_pass=low_pass,
            high_pass=high_pass,
        )[padlen:-padlen, :]  # remove 0-pad frames on both sides
        assert filtered_data.shape[0] == func_data.shape[0], "Filtered data must have the same number of timepoints as the original functional data"
    else:
        filtered_data = butterworth(
            signals=padded_func_data,
            sampling_rate=1.0 / tr,
            low_pass=low_pass,
            high_pass=high_pass,
            padtype=None if padtype == "none" else padtype,
            padlen=padlen
        )

    return filtered_data


@debug_logging
def create_final_design(data_list: list[npt.ArrayLike],
                        design_list: list[tuple[pd.DataFrame, int]],
                        noise_list: list[pd.DataFrame] = None,
                        exclude_global_mean: bool = False):
    """
    Creates a final, concatenated design matrix for all functional runs in a session

    Parameters
    ----------

    data_list: list[npt.ArrayLike]
        List of numpy arrays representing BOLD data
    design_list: list[pd.DataFrame]
        List of created design matrices corresponding to each respective BOLD run in data_list
    noise_list: list[pd.DataFrame]
        List of DataFrame objects corresponding to models of noise for each respective BOLD run in data_list
    exclude_global_mean: bool
        Flag to indicate that a global mean should not be included into the final design matrix

    Returns
    -------

    Returns a tuple containing the final concatenated data in index 0, and the
    final concatenated design matrix in index 1.
    """
    num_runs = len(data_list)
    assert num_runs == len(design_list), "There should be the same number of design matrices and functional runs"

    design_df_list = [t[0] for t in design_list]
    if noise_list:
        assert num_runs == len(noise_list), "There should be the same number of noise matrices and functional runs"
        for i in range(num_runs):
            noise_df = noise_list[i]
            assert len(noise_df) == len(design_df_list[i])
            run_num = design_list[i][1]
            rename_dict = dict()
            for c in noise_df.columns:
                if ("trend" in c) or ("mean" in c) or ("spike" in c):
                    rename_dict[c] = f"run-{run_num:02d}_{c}"
            noise_df = noise_df.rename(columns=rename_dict)
            noise_list[i] = noise_df
            design_df_list[i] = pd.concat([design_df_list[i].reset_index(drop=True), noise_df.reset_index(drop=True)], axis=1)

    final_design = pd.concat(design_df_list, axis=0, ignore_index=True).fillna(0)
    if not exclude_global_mean:
        final_design.loc[:, "global_mean"] = 1
    final_data = np.concat(data_list, axis=0)
    return (final_data, final_design)


@debug_logging
def massuni_linGLM(func_data: npt.ArrayLike,
                   design_matrix: pd.DataFrame,
                   mask: npt.ArrayLike,
                   stdscale: bool):
    """
    Compute the mass univariate GLM.

    Parameters
    ----------

    func_data: npt.ArrayLike
        Numpy array representing BOLD data
    design_matrix: pd.DataFrame
        DataFrame representing a design matrix for the GLM
    mask: npt.ArrayLike
        Numpy array representing a mask to apply to the two other parameters
    """
    assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"
    assert mask.dtype == bool

    # apply the mask to the data
    design_matrix = design_matrix.to_numpy()

    func_ss = StandardScaler()
    design_ss = StandardScaler()
    if stdscale:
        masked_func_data = func_ss.fit_transform(func_data.copy()[mask, :])
        masked_design_matrix = design_ss.fit_transform(design_matrix.copy()[mask, :])
    else:
        masked_func_data = func_data.copy()[mask, :]
        masked_design_matrix = design_matrix.copy()[mask, :]

    # standardize the masked data

    # comput beta values
    inv_mat = np.linalg.pinv(masked_design_matrix)
    beta_data = np.dot(inv_mat, masked_func_data)

    # standardize the unmasked data
    if stdscale:
        func_data = func_ss.transform(func_data)
        design_matrix = design_ss.transform(design_matrix)

    # compute the residuals with unmasked data
    est_values = np.dot(design_matrix, beta_data)
    resids = func_data - est_values

    return (beta_data, resids)


def autogenerate_mask(mask_files: list[Path], output_path: Path) -> Path:
    logger.info(f"Autogenerating mask at {output_path}...")
    if len(mask_files) == 1:
        return mask_files[0]
    orig_img = nib.load(mask_files[0], mmap=False)
    fdata = orig_img.get_fdata()
    for mask_file in mask_files[1:]:
        fdata += nib.load(mask_file, mmap=False).get_fdata()
    fdata[fdata >= 1] = 1
    new_img = nib.nifti1.Nifti1Image(
        fdata,
        affine=orig_img.affine,
        header=orig_img.header
    )
    nib.save(new_img, output_path)
    return output_path


def get_parser():
    parser = OceanParser(
        prog="oceanfla",
        description="Ocean Labs first level analysis",
        fromfile_prefix_chars="@",
        epilog="An arguments file can be accepted with @FILEPATH"
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    session_arguments = parser.add_argument_group("Session Specific")
    config_arguments = parser.add_argument_group("Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")

    session_arguments.add_argument("--subject", "-su", required=True,
                                   help="The subject ID")
    session_arguments.add_argument("--session", "-se", required=True,
                                   help="The session ID")
    session_arguments.add_argument("--events_long", "-el", type=Path, nargs="?", const=lambda a: a.derivs_dir / a.preproc_subfolder,
                                   help="""Path to the directory containing long formatted event files to use.
                        Default is the derivatives directory containing preprocessed outputs.""")
    session_arguments.add_argument("--export_args", "-ea", type=Path,
                                   help="Path to a file to save the current arguments.")
    session_arguments.add_argument("--force_overwrite", action="store_true",
                                   help="Use this flag to force oceanfla to proceed when conflicting task outputs are present in the output directory")
    session_arguments.add_argument("--debug", action="store_true",
                                   help="Use this flag to save intermediate outputs for a chance to debug inputs")

    config_arguments.add_argument("--task", "-t", required=True,
                                  help="The name of the task to analyze.")
    config_arguments.add_argument("--bold_file_type", "-ft", required=True,
                                  help="The file type of the functional runs to use.")
    config_arguments.add_argument("--brain_mask", "-bm", type=Path,
                                  help="If the bold file type is volumetric data, a brain mask must also be supplied.")
    config_arguments.add_argument("--func_space",
                                  help="Space that the preprocessed data should be in (for example, 'T2w', 'MNIInfant', etc.)")
    config_arguments.add_argument("--fwhm",
                                  help="FWHM smoothing kernel, in mm (only applies to CIFTI data)",
                                  type=float)
    config_arguments.add_argument("--derivs_dir", "-d", type=Path, required=True,
                                  help="Path to the BIDS formatted derivatives directory containing processed outputs.")
    config_arguments.add_argument("--preproc_subfolder", "-pd", type=str, default="fmriprep",
                                  help="Name of the subfolder in the derivatives directory containing the preprocessed bold data. Default is 'fmriprep'")
    config_arguments.add_argument("--raw_bids", "-r", type=Path,
                                  help="Path to the BIDS formatted raw data directory for this dataset.")
    config_arguments.add_argument("--derivs_subfolder", "-ds", default="first_level",
                                  help="The name of the subfolder in the derivatives directory where bids style outputs should be stored. The default is 'first_level'.")
    config_arguments.add_argument("--output_dir", "-o", type=Path,
                                  help="Alternate Path to a directory to store the results of this analysis. Default is '[derivs_dir]/first_level/'")
    config_arguments.add_argument("--custom_desc", "-cd",
                                  help="A custom description to add in the file name of every output file.")
    config_arguments.add_argument("--fir", "-ff", type=int,
                                  help="The number of frames to use in an FIR model.")
    config_arguments.add_argument("--fir_vars", nargs="*",
                                  help="""A list of the task regressors to apply this FIR model to. The default is to apply it to all regressors if no
                        value is specified. A list must be specified if both types of models are being used""")
    config_arguments.add_argument("--hrf", nargs=2, type=int, metavar=("PEAK", "UNDER"),
                                  help="""Two values to describe the hrf function that will be convolved with the task events.
                        The first value is the time to the peak, and the second is the undershoot duration. Both in units of seconds.""")
    config_arguments.add_argument("--hrf_vars", nargs="*",
                                  help="""A list of the task regressors to apply this HRF model to. The default is to apply it to all regressors if no
                        value is specifed. A list must be specified if both types of models are being used.""")
    config_arguments.add_argument("--custom_hrf", "-ch", type=Path,
                                  help="The path to a txt file containing the timeseries for a custom hrf to use instead of the double gamma hrf")
    config_arguments.add_argument("--unmodeled", "-um", nargs="*",
                                  help="""A list of the task regressors to leave unmodeled, but still included in the final design matrix. These are
                        typically continuous variables that need not be modeled with hrf or fir, but any of the task regressors can be included.""")
    config_arguments.add_argument("--start_censoring", "-sc", type=int, default=0,
                                  help="The number of frames to censor out at the beginning of each run. Typically used to censor scanner equilibrium time. Default is 0")
    config_arguments.add_argument("--confounds", "-c", nargs="+", default=[],
                                  help="A list of confounds to include from each confound timeseries tsv file.")
    config_arguments.add_argument("--fd_threshold", "-fd", type=float, default=0.9,
                                  help="The framewise displacement threshold used when censoring high-motion frames")
    config_arguments.add_argument("--minimum_unmasked_neighbors", type=int, default=None,
                                  help="Minimum number of contiguous unmasked frames on either side of a given frame that's required to be under the fd_threshold; any unmasked frame without the required number of neighbors will be masked.")
    config_arguments.add_argument("--tmask", action=argparse.BooleanOptionalAction,
                                  help="Flag to indicate that tmask files, if found with the preprocessed outputs, should be used. Tmask files will override framewise displacement threshold censoring if applicable.")
    config_arguments.add_argument("--repetition_time", "-tr", type=float,
                                  help="Repetition time of the function runs in seconds. If it is not supplied, an attempt will be made to read it from the JSON sidecar file.")
    config_arguments.add_argument("--detrend_data", "-dd", action="store_true",
                                  help="""Flag to demean and detrend the data before modeling. The default is to include
                        a mean and trend line into the nuisance matrix instead.""")
    config_arguments.add_argument("--percent_change", "-pc", action="store_true",
                                  help="""Flag convert data to percent signal change.""")
    config_arguments.add_argument("--no_global_mean", action="store_true",
                                  help="Flag to indicate that you do not want to include a global mean into the model.")
    high_motion_params = config_arguments.add_mutually_exclusive_group()
    high_motion_params.add_argument("--spike_regression", "-sr", action="store_true",
                                    help="Flag to indicate that framewise displacement spike regression should be included in the nuisance matrix.")
    high_motion_params.add_argument("--fd_censoring", "-fc", action="store_true",
                                    help="Flag to indicate that frames above the framewise displacement threshold should be censored before the GLM.")
    config_arguments.add_argument("--run_exclusion_threshold", "-re", type=int,
                                  help="The percent of frames a run must retain after high motion censoring to be included in the fine GLM. Only has effect when '--fd_censoring' is active.")
    config_arguments.add_argument("--min_average_tsnr", type=float, default=0,
                                  help="The minimum whole-brain-average TSNR (across unmasked frames) required for a run to be included in analysis.")
    config_arguments.add_argument("--min_run_threshold", type=int, default=0,
                                  help="The minimum number of unexcluded runs required to run the GLM")
    config_arguments.add_argument("--nuisance_regression", "-nr", nargs="*", default=[],
                                  help="""List of variables to include in nuisance regression before the performing the GLM for event-related activation. If no values are specified then
                                  all nuisance/confound variables will be included""")
    config_arguments.add_argument("--nuisance_fd", "-nf", type=float,
                                  help="The framewise displacement threshold used when censoring frames for nuisance regression.")
    config_arguments.add_argument("--highpass", "-hp", type=float, nargs="?", const=0.008,
                                  help="""The high pass cutoff frequency for signal filtering. Frequencies below this value (Hz) will be filtered out. If the argument
                        is supplied but no value is given, then the value will default to 0.008 Hz""")
    config_arguments.add_argument("--lowpass", "-lp", type=float, nargs="?", const=0.1,
                                  help="""The low pass cutoff frequency for signal filtering. Frequencies above this value (Hz) will be filtered out. If the argument
                        is supplied but no value is given, then the value will default to 0.1 Hz""")
    config_arguments.add_argument("--classic_fd", action="store_true",
                                  help="Search for 'classic' .FD files instead of .tsv files for confounds (only difference in naming is the suffix)")
    config_arguments.add_argument("--filter_padtype", default="zero",
                                  choices=["odd", "even", "zero", "constant", "none"],
                                  help="Type of padding to use for low-, high-, or band-pass filter, if one is applied.")
    config_arguments.add_argument("--filter_padlen", type=int, default=50,
                                  help="Length of padding to add to the beginning and end of BOLD run before applying butterworth filter.")
    config_arguments.add_argument("--volterra_lag", "-vl", nargs="?", const=2, type=int,
                                  help="""The amount of frames to lag for a volterra expansion. If no value is specified
                        the default of 2 will be used. Must be specifed with the '--volterra_columns' option.""")
    config_arguments.add_argument("--volterra_columns", "-vc", nargs="+", default=[],
                                  help="The confound columns to include in the expansion. Must be specifed with the '--volterra_lag' option.")
    config_arguments.add_argument("--parcellate", "-parc", type=Path,
                                  help="Path to a dlabel file to use for parcellation of a dtseries")
    config_arguments.add_argument("--stdscale_glm", choices=["runlevel", "seslevel", "both", "none"], default="seslevel",
                                  help="Option to standard scale concatenated timeseries before running final GLM (after masking & nuisance regression)")
    config_arguments.add_argument("--np_random_seed", type=int, default=42,
                                  help="Set Numpy seed (default is 42)")
    return (parser, config_arguments)


def main():
    parser, config_arguments = get_parser()
    args = parser.parse_args()

    np.random.default_rng(seed=args.np_random_seed)

    if args.hrf is not None and args.fir is not None:
        if not args.fir_vars or not args.hrf_vars:
            parser.error("Must specify variables to apply each model to if using both types of models")
    elif args.hrf is None and args.fir is None:
        parser.error("Must include model parameters for at least one of the models, fir or hrf.")

    if args.custom_hrf:
        if not (args.custom_hrf.exists() and args.custom_hrf.suffix == ".txt"):
            parser.error("The 'custom_hrf' argument must be a file of type '.txt' and must exist")

    if args.bold_file_type[0] != ".":
        args.bold_file_type = "." + args.bold_file_type
    if args.bold_file_type == ".nii" or args.bold_file_type == ".nii.gz":
        imagetype = "nifti"
    else:
        imagetype = "cifti"

    if args.parcellate:
        if (not args.parcellate.exists()) or (not args.parcellate.name.endswith(".dlabel.nii")):
            parser.error("The 'parcellate' argument must be a file of type '.dlabel.nii' and must exist")

    flags.parcellated = (args.parcellate or args.bold_file_type == ".ptseries.nii")

    if (args.volterra_lag and not args.volterra_columns) or (not args.volterra_lag and args.volterra_columns):
        parser.error("The options '--volterra_lag' and '--volterra_columns' must be specifed together, or neither of them specified.")

    if callable(args.events_long):
        args.events_long = args.events_long(args)

    try:
        assert args.derivs_dir.is_dir(), "Derivatives directory must exist but is not found"
    except AssertionError as e:
        logger.exception(e)
        exit_program_early(e)

    # Export the current arguments to a file
    if args.export_args:
        try:
            assert args.export_args.parent.exists() and args.export_args.suffix, "Argument export path must be a file path in a directory that exists"
            log_linebreak()
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######\n")
            export_args_to_file(args, config_arguments, args.export_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    user_desc = f"-{args.custom_desc}" if args.custom_desc else ""
    file_name_base = f"sub-{args.subject}_ses-{args.session}_task-{args.task}"

    if not hasattr(args, "output_dir") or args.output_dir is None:
        args.output_dir = args.derivs_dir / f"{args.derivs_subfolder}/sub-{args.subject}/ses-{args.session}/func"

    # check if previous outputs exist in the output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_description_json = args.derivs_dir / args.derivs_subfolder / "dataset_description.json"
    descriptions_list = []
    descriptions_tsv = args.derivs_dir / args.derivs_subfolder / "descriptions.tsv"
    unmodified_output_dir_contents = set(args.output_dir.iterdir())
    old_outputs = set(args.output_dir.glob(f"{file_name_base}*"))
    if not args.force_overwrite and len(old_outputs) != 0:
        want_to_delete = prompt_user_continue(dedent(f"""
            The output directory for this subject and session contain derivative files for this task: {args.task}
            Would you like to delete these files and start fresh? If not, the program will exit now.
            """))
        if want_to_delete:
            for old_file in old_outputs:
                logger.info(f"deleting file: {old_file.resolve()}")
                os.remove(old_file)
                unmodified_output_dir_contents.discard(old_file)
        else:
            exit_program_early("output directory will not be modified")

    # set up the logging
    log_dir = args.output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{file_name_base}_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}{user_desc}.log"
    add_file_handler(logger, log_path)

    if args.debug:
        flags.debug = True
        logger.setLevel(logging.DEBUG)

    logger.info("Starting oceanfla...")
    logger.info(f"Log will be stored at {log_path}")

    # log the arguments used for this run
    for k,v in (dict(args._get_kwargs())).items():
        logger.info(f" {k} : {v}")

    model_type = "Mixed" if args.fir and args.hrf else "FIR" if args.fir else "HRF"
    file_map_list = []

    log_stats = "\nRun number\tTotal frames\tFrames cesnored\n"

    try:
        # find all preprocessed BOLD runs for this subject and session
        preproc_derivs = args.derivs_dir / args.preproc_subfolder / f"sub-{args.subject}"
        if args.func_space:
            bold_files = sorted(preproc_derivs.glob(f"**/sub-{args.subject}_ses-{args.session}*task-{args.task}*space-{args.func_space}*bold{args.bold_file_type}"))
            mask_files = sorted(preproc_derivs.glob(f"**/sub-{args.subject}_ses-{args.session}*task-{args.task}*space-{args.func_space}*desc-brain_mask{args.bold_file_type}"))
        else:
            bold_files = sorted(preproc_derivs.glob(f"**/sub-{args.subject}_ses-{args.session}*task-{args.task}*bold{args.bold_file_type}"))
            mask_files = sorted(preproc_derivs.glob(f"**/sub-{args.subject}_ses-{args.session}*task-{args.task}*desc-brain_mask{args.bold_file_type}"))
        assert len(bold_files) > 0, "Did not find any bold files in the given derivatives directory for the specified task and file type"

        if args.parcellate and imagetype == "cifti":
            bold_files = [parcellate_dtseries(dtseries_path=dt, parc_dlabel_path=args.parcellate) for dt in bold_files]
            args.bold_file_type = ".ptseries.nii"

        brain_mask = args.brain_mask
        if imagetype == "nifti" and (not brain_mask or not brain_mask.is_file()):
            if args.func_space:
                brain_mask = bold_files[0].parent / f"sub-{args.subject}_ses-{args.session}_task-{args.task}_space-{args.func_space}_desc-masterbrain_mask{args.bold_file_type}"
            else:
                brain_mask = bold_files[0].parent / f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-masterbrain_mask{args.bold_file_type}"
            if not brain_mask.is_file():
                autogenerate_mask(mask_files, brain_mask)

        # for each BOLD run, find the accompanying confounds file and events/events long file
        for bold_path in bold_files:
            file_map = {"bold": bold_path}
            bold_base = bold_path.name.split("_space")[0]
            bold_base = bold_base.split("_desc")[0]

            confounds_suffix = "FD" if args.classic_fd else "[ct]sv"
            confounds_search_path = f"{bold_base}_desc*-confounds_timeseries.{confounds_suffix}"
            confounds_files = list(bold_path.parent.glob(confounds_search_path))
            assert len(confounds_files) == 1, f"Found {len(confounds_files)} confounds files for bold run: {str(bold_path)} search path: {confounds_search_path}"
            if args.classic_fd:  # Make .tsv file out of .FD file (this just adds columns and removes first derivative, may change this behavior in the future)
                fd_df = pd.read_csv(
                    os.readlink(confounds_files[0]) if op.islink(confounds_files[0]) else confounds_files[0],
                    sep="\t",
                    header=None
                )
                fd_df.drop(1, axis=1, inplace=True)  # drop first derivative in second column
                fd_df.rename(columns={0: "framewise_displacement"}, inplace=True)
                fd_df.to_csv(
                    converted_fd_tsv := Path(str(confounds_files[0]).replace('.FD', '.tsv')),
                    sep="\t"
                )
                file_map["confounds"] = converted_fd_tsv
            else:
                file_map["confounds"] = confounds_files[0] if not op.islink(confounds_files[0]) else os.readlink(confounds_files[0])

            if args.tmask:
                tmask_search_path = f"{bold_base}*_tmask.txt"
                tmask_files = list(bold_path.parent.glob(tmask_search_path))
                if len(tmask_files) != 1:
                    logger.warning(f"Found {len(tmask_files)} tmask files for bold run: {str(bold_path)} search path: {tmask_search_path}. No tmask will be used for this run.")
                else:
                    file_map["tmask"] = tmask_files[0]

            event_search_paths = [preproc_derivs]
            if args.raw_bids:
                event_search_paths.insert(0, args.raw_bids)
            if args.events_long:
                event_search_paths.insert(0, args.events_long)
            file_map["events"] = None
            for idx, event_search_path in enumerate(event_search_paths):
                event_glob = f"{bold_base}*_events{'_long' if args.events_long else ''}.[ct]sv"
                event_files = list(event_search_path.rglob(event_glob))
                logger.info(f"Found {len(event_files)} event files for bold run: {str(bold_path)} search path: {str(event_search_path) + '/**/' + str(event_glob)}")
                if not len(event_files) == 1:
                    continue
                file_map["events"] = event_files[0]
                break
            if file_map["events"] is None:
                raise FileNotFoundError(dedent(f"""
                    Found no event files in these directories:
                    {', '.join([str(p) for p in event_search_paths])}
                    {'(Maybe you forgot --raw_bids?)' if not args.raw_bids else ''}
                """))

            file_map_list.append(file_map)

        tr = args.repetition_time
        img_header = None
        trial_types = set()
        func_data_list = []
        design_df_list = []
        noise_df_list = []
        mask_list = []

        # For each set of run files, create the design matrix and finalize the BOLD data before the session-task level GLM
        for map_dex, run_map in enumerate(file_map_list):
            log_linebreak()
            logger.info(f"processing bold file: {run_map['bold']}")
            logger.info("loading in BOLD data")
            run_file_base = file_name_base
            run_info = len(str(run_map['bold']).split('run-')) > 1
            run_num = map_dex + 1
            if run_info:
                run_num = int(run_map['bold'].name.split('run-')[-1].split('_')[0])
                run_file_base = f"{file_name_base}_run-{run_num:02d}"

            func_data, read_tr, read_header = load_data(
                func_file=run_map['bold'],
                brain_mask=brain_mask,
                need_tr=(not tr),
                fwhm=args.fwhm
            )

            tr = tr if tr else read_tr
            img_header = img_header if img_header else read_header

            step_count = 0
            # create the events matrix
            logger.info(" reading events file and creating design matrix")
            events_long = None
            if args.events_long:
                events_long = pd.read_csv(
                    run_map["events"],
                    index_col=0,
                    sep="," if str(run_map["events"]).endswith('.csv') else "\t")
            else:
                events_long = make_events_long(
                    event_file=run_map["events"],
                    volumes=func_data.shape[0],
                    tr=tr,
                )
            events_df, run_conditions = events_to_design(
                events_long=events_long,
                fir=args.fir,
                fir_list=args.fir_vars if args.fir_vars else None,
                hrf=args.custom_hrf if args.custom_hrf else args.hrf,
                hrf_list=args.hrf_vars if args.hrf_vars else None,
                unmodeled_list=args.unmodeled
            )

            # create the noise matrix
            logger.info(" reading confounds file and creating nuisance matrix")
            noise_df = make_noise_ts(
                confounds_file=run_map["confounds"],
                confounds_columns=args.confounds,
                # demean=(args.highpass is not None or args.lowpass is not None or "mean" in args.nuisance_regression),
                # linear_trend=(args.highpass is not None or args.lowpass is not None or "trend" in args.nuisance_regression),
                demean=(not args.detrend),
                linear_trend=(not args.detrend),
                spike_threshold=args.fd_threshold if args.spike_regression else None,
                volterra_expansion=args.volterra_lag,
                volterra_columns=args.volterra_columns
            )

            # create and apply the acqusition mask
            acquisition_mask = np.ones(shape=(func_data.shape[0],)).astype(bool)
            acquisition_mask[:args.start_censoring] = False
            if args.start_censoring > 0:
                logger.info(f" removing the first {args.start_censoring} frames from the beginning of the run")
            func_data = func_data[acquisition_mask, :]
            events_df = events_df.loc[acquisition_mask, :]
            noise_df = noise_df.loc[acquisition_mask, :]

            # create high motion mask and exclude run if needed
            run_mask = np.full((func_data.shape[0],), 1).astype(bool)

            if args.tmask or args.fd_censoring:
                if "tmask" in run_map:
                    logger.info(f" censoring timepoints using the tmask file: {run_map['tmask']}")
                    tmask = np.loadtxt(run_map["tmask"], dtype=int).astype(bool)
                    assert tmask.shape == acquisition_mask.shape, f"Tmask file ({tmask.shape[0]}) does not match the length of the run ({acquisition_mask.shape[0]}): {run_map['tmask']}"
                    run_mask &= tmask[acquisition_mask]

                elif args.fd_censoring:
                    logger.info(f" censoring timepoints using a high motion mask with a framewise displacement threshold of {args.fd_threshold}")
                    confounds_df = pd.read_csv(run_map["confounds"], sep="," if str(run_map["confounds"]).endswith(".csv") else "\t")
                    fd_arr = confounds_df.loc[:, "framewise_displacement"].to_numpy()[acquisition_mask]
                    if args.minimum_unmasked_neighbors:
                        fd_arr_padded = np.pad(fd_arr, pad_width := args.minimum_unmasked_neighbors)
                        fd_mask = np.full(fd_arr_padded.shape, False)
                        for i in range(pad_width, len(fd_arr_padded) - pad_width):
                            if all(fd_arr_padded[range(i - pad_width, i + pad_width + 1)] < args.fd_threshold):
                                fd_mask[i] = True
                            elif i - pad_width < pad_width and all(fd_arr_padded[range(pad_width, i + pad_width + 1)] < args.fd_threshold):
                                fd_mask[i] = True
                            elif i + pad_width + 1 > len(fd_arr_padded) - pad_width and all(fd_arr_padded[range(i - pad_width, len(fd_arr_padded) - pad_width)] < args.fd_threshold):
                                fd_mask[i] = True
                            else:
                                fd_mask[i] = False
                        fd_mask = fd_mask[pad_width:-pad_width]
                    else:
                        fd_mask = fd_arr < args.fd_threshold
                    run_mask &= fd_mask
                logger.info(f" a total of {np.sum(~run_mask)} timepoints will be censored from this run")
                log_stats += f"{run_num:02d}\t{run_mask.shape[0]}\t{np.sum(~run_mask)}\n"
                frame_retention_percent = (np.sum(run_mask) / run_mask.shape[0]) * 100

                # if censoring causes the number of retained frames to be below the run exclusion threshold, drop the run
                if args.run_exclusion_threshold and (frame_retention_percent < args.run_exclusion_threshold):
                    logger.info(f" BOLD run: {run_map['bold']} has fell below the run exclusion threshold of {args.run_exclusion_threshold}% and will not be used in the final GLM.")
                    continue
                run_tsnr = avg_tsnr(func_data[run_mask, :])
                if args.min_average_tsnr and run_tsnr < args.min_average_tsnr:
                    logger.info(f" BOLD run: {run_map['bold']} has average TSNR {run_tsnr} below the average TSNR exclusion threshold of {args.min_average_tsnr} and will not be used in the final GLM.")
                    continue

            # save out the nuisance matrix and events matrix (if debug)
            noise_df_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}_nuisance.csv"
            logger.info(f" saving nuisance matrix to file: {noise_df_filename}")
            noise_df.to_csv(noise_df_filename)
            unmodified_output_dir_contents.discard(noise_df_filename)

            events_long_filename = args.output_dir / f"{run_file_base}_desc{user_desc}-events_long.csv"
            logger.debug(f" saving events long to file: {events_long_filename}")
            events_long.to_csv(events_long_filename)
            unmodified_output_dir_contents.discard(events_long_filename)

            if flags.debug:
                events_df_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}-events_matrix.csv"
                logger.debug(f" saving events matrix to file: {events_df_filename}")
                events_df.to_csv(events_df_filename)
                unmodified_output_dir_contents.discard(events_df_filename)

            if args.percent_change:
                func_data_psc = percent_signal_change(func_data, run_mask)
                run_map["data_psc"] = func_data_psc
                func_data = func_data_psc
                step_count += 1
                if flags.debug:
                    pscimg, img_suffix = create_image(
                        data=func_data,
                        imagetype=imagetype,
                        brain_mask=brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    cleaned_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}_step-{step_count}_psc{img_suffix}"
                    logger.debug(f" saving BOLD data after detrending to file: {cleaned_filename}")
                    nib.save(
                        pscimg,
                        cleaned_filename
                    )
                    unmodified_output_dir_contents.discard(cleaned_filename)

            # detrend the BOLD data if specifed
            if args.detrend_data:
                logger.info(" detrending the BOLD data")
                func_data_detrend = demean_detrend(
                    func_data=func_data
                )
                run_map["data_detrend"] = func_data_detrend
                func_data = func_data_detrend
                if flags.debug:
                    cleanimg, img_suffix = create_image(
                        data=func_data,
                        imagetype=imagetype,
                        brain_mask=brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    cleaned_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}_detrended{img_suffix}"
                    logger.debug(f" saving BOLD data after detrending to file: {cleaned_filename}")
                    nib.save(
                        cleanimg,
                        cleaned_filename
                    )
                    unmodified_output_dir_contents.discard(cleaned_filename)

            # demean and detrend if filtering is requested
            if (args.highpass is not None or args.lowpass is not None):
                if "mean" not in args.nuisance_regression:
                    logger.warning("High-, low-, or band-pass specified, but mean not specified as a nuisance regressor -- adding this in automatically")
                    args.nuisance_regression.append("mean")
                if "trend" not in args.nuisance_regression:
                    logger.warning("High-, low-, or band-pass specified, but mean not specified as a nuisance regressor -- adding this in automatically")
                    args.nuisance_regression.append("trend")

            # nuisance regression if specified
            if len(args.nuisance_regression) > 0:
                nuisance_mask = np.ones(shape=(func_data.shape[0],)).astype(bool)
                if args.nuisance_fd:
                    logger.info(f" censoring timepoints for nuisance regression using the framewise displacement threshold of {args.nuisance_fd}")
                    confounds_df = pd.read_csv(run_map["confounds"], sep="," if str(run_map["confounds"]).endswith(".csv") else "\t")
                    nuisance_fd_arr = confounds_df.loc[:, "framewise_displacement"].to_numpy()[acquisition_mask]
                    if args.minimum_unmasked_neighbors:
                        nuisance_fd_arr_padded = np.pad(nuisance_fd_arr, pad_width := args.minimum_unmasked_neighbors)
                        nuisance_fd_mask = np.full(nuisance_fd_arr_padded.shape, False)
                        for i in range(pad_width, len(nuisance_fd_arr_padded) - pad_width):
                            if all(nuisance_fd_arr_padded[range(i - pad_width, i + pad_width + 1)] < args.nuisance_fd):
                                nuisance_fd_mask[i] = True
                            else:
                                nuisance_fd_mask[i] = False
                        nuisance_fd_mask = nuisance_fd_mask[pad_width:-pad_width]
                    else:
                        nuisance_fd_mask = nuisance_fd_arr < args.nuisance_fd
                    nuisance_mask &= nuisance_fd_mask
                    logger.info(f" a total of {np.sum(~nuisance_mask)} timepoints will be censored with this nuisance framewise displacement threshold")

                noise_columns = noise_df.columns.to_list()
                not_found_noise_vars = {c for c in args.nuisance_regression if c not in noise_columns}
                if len(not_found_noise_vars) > 0:
                    logger.info(f"The following nuisance variables were not found in the nuisance matrix and will not be used for nuisance regression: {','.join(not_found_noise_vars)}")
                args.nuisance_regression = list({c for c in args.nuisance_regression if c in noise_columns}) if len(args.nuisance_regression) > 0 else noise_columns
                if "mean" not in args.nuisance_regression:
                    args.nuisance_regression.append("mean")

                leftover_noise_columns = {c for c in noise_columns if c not in args.nuisance_regression}
                leftover_noise_columns.update(["mean", "trend"])
                noise_regression_df = noise_df.loc[:, args.nuisance_regression].copy()
                noise_df = noise_df.loc[:, list(leftover_noise_columns)].copy() if len(leftover_noise_columns) > 0 else None

                logger.info(" performing nuisance regression")
                nuisance_betas, func_data_residuals = massuni_linGLM(
                    func_data=func_data,
                    design_matrix=noise_regression_df,
                    mask=nuisance_mask,
                    stdscale=args.stdscale_glm in ("both", "runlevel")
                )

                run_map["data_resids"] = func_data_residuals
                func_data = func_data_residuals
                step_count += 1
                if flags.debug:
                    # save out the BOLD data after nuisance regression
                    nrimg, img_suffix = create_image(
                        data=func_data,
                        imagetype=imagetype,
                        brain_mask=brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    nr_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}_step-{step_count}_nuisance-regressed{img_suffix}"
                    logger.debug(f" saving BOLD data after nuisance regression to file: {nr_filename}")
                    nib.save(
                        nrimg,
                        nr_filename
                    )
                    unmodified_output_dir_contents.discard(nr_filename)

                    # save out the nuisance betas
                    for i, noise_col in enumerate(noise_regression_df.columns):
                        beta_img, img_suffix = create_image(
                            data=np.expand_dims(nuisance_betas[i,:], axis=0),
                            imagetype=imagetype,
                            brain_mask=brain_mask,
                            tr=tr,
                            header=img_header
                        )
                        beta_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}-beta-{noise_col}-frame-0{user_desc}{img_suffix}"
                        logger.debug(f" saving betas for nuisance variable: {noise_col} to file: {beta_filename}")
                        nib.save(
                            beta_img,
                            beta_filename
                        )
                        unmodified_output_dir_contents.discard(beta_filename)

            # filter the data if specified
            if args.lowpass or args.highpass:
                logger.info(f" filtering the BOLD data with a highpass of {args.highpass} and a lowpass of {args.lowpass}")
                func_data_filtered = filter_data(
                    func_data=func_data,
                    mask=run_mask,
                    tr=tr,
                    low_pass=args.lowpass if args.lowpass else None,
                    high_pass=args.highpass if args.highpass else None,
                    padtype=args.filter_padtype,
                    padlen=args.filter_padlen,
                )
                run_map["data_filtered"] = func_data_filtered
                func_data = func_data_filtered
                step_count += 1
                if flags.debug:
                    cleanimg, img_suffix = create_image(
                        data=func_data,
                        imagetype=imagetype,
                        brain_mask=brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    filtered_filename = args.output_dir / f"{run_file_base}_desc-model-{model_type}{user_desc}_step-{step_count}_filtered{img_suffix}"
                    logger.debug(f" saving BOLD data after filtering to file: {filtered_filename}")
                    nib.save(
                        cleanimg,
                        filtered_filename
                    )
                    unmodified_output_dir_contents.discard(filtered_filename)

            # apppend the run-wise data to the session list
            if noise_df is not None:
                assert func_data.shape[0] == len(noise_df), "The functional data and the nuisance matrix have a different number of timepoints"
                logger.info(" appending the nuisance matrix to the run list")
                noise_df_list.append(noise_df)

            logger.info(" appending the high motion mask to the run list")
            assert func_data.shape[0] == run_mask.shape[0], "The functional data and the high motion mask have a different number of timepoints"
            mask_list.append(run_mask)

            logger.info(" appending the BOLD data and design matrix to the run list")
            trial_types.update(run_conditions)
            assert func_data.shape[0] == len(events_df), "The functional data and the design matrix have a different number of timepoints"
            func_data_list.append(func_data)
            design_df_list.append((events_df,run_num))
            logger.info(f" a total of {np.sum(run_mask)} frames will be used from BOLD file: {run_map['bold']}")

        log_linebreak()
        # check that at least one run passed the run exculsion threshold
        assert len(func_data_list) == len(design_df_list) and (len(noise_df_list) == 0 or (len(noise_df_list) == len(func_data_list))), "Something went wrong! Run-wise data lists are not the same size."
        if len(func_data_list) == 0:
            logger.info(f"all run have been excluded with the run exclusion threshold of {args.run_exclusion_threshold}%, try lowering this parameter before running again!")
            return
        elif len(func_data_list) < args.min_run_threshold:
            logger.info(f"At least {args.min_run_threshold} runs must be present to run the GLM")
            return

        # create the final design matrix and append the data together
        logger.info("concatenating run level BOLD data and design matrices for GLM")
        final_func_data, final_design_unmasked = create_final_design(
            data_list=func_data_list,
            design_list=design_df_list,
            noise_list=noise_df_list if len(noise_df_list) == len(func_data_list) else None,
            exclude_global_mean=args.no_global_mean
        )
        final_high_motion_mask = np.concat(mask_list, axis=0)
        logger.info(f"total number of framess after start censoring for each run: {final_high_motion_mask.shape[0]}")
        logger.info(f"total number of frames that will be used in the final GLM after high motion censoring: {np.sum(final_high_motion_mask)}")

        final_design_masked = final_design_unmasked.loc[final_high_motion_mask, :]
        final_design_unmasked_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}{user_desc}-design_unmasked.csv"
        logger.info(f"saving the final unmasked design matrix to file: {final_design_unmasked_filename}")
        final_design_unmasked.to_csv(final_design_unmasked_filename)
        unmodified_output_dir_contents.discard(final_design_unmasked_filename)
        final_design_masked_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}{user_desc}-design_final.csv"
        logger.info(f"saving the final design matrix to file: {final_design_masked_filename}")
        final_design_masked.to_csv(final_design_masked_filename)
        unmodified_output_dir_contents.discard(final_design_masked_filename)

        # run the final GLM
        logger.info("running GLM on concatenated BOLD data with final design matrix")

        activation_betas, func_residual = massuni_linGLM(
            func_data=final_func_data,
            design_matrix=final_design_unmasked,
            mask=final_high_motion_mask,
            stdscale=args.stdscale_glm in ("seslevel", "both")
        )
        step_count += 1
        # save out the beta values
        logger.info("saving betas from GLM into files")
        fir_betas_to_combine = set()
        for i, c in enumerate(final_design_unmasked.columns):
            if args.fir and c[-3] == "_" and c[-2:].isnumeric() and c[:-3] in trial_types:
                fir_betas_to_combine.add(c[:-3])
                continue
            elif c in trial_types:
                beta_img, img_suffix = create_image(
                    data=np.expand_dims(activation_betas[i,:], axis=0),
                    imagetype=imagetype,
                    brain_mask=brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}-beta-{c}-frame-0{user_desc}{img_suffix}"
                descriptions_list.append({
                    "desc_id": f"model-{model_type}-beta-{c}-frame-0{user_desc}",
                    "description": f"Represents beta weights in a {model_type} model, representing modelled regressor {c}. {'Additional notes: ' + user_desc if len(user_desc) > 0 else ''}",
                    "model_type": model_type,
                    "condition": c,
                    "additional_desc": user_desc,
                    "frame": 0,
                    "is_nuisance": False
                })
                logger.info(f" saving betas for variable {c} to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )
                unmodified_output_dir_contents.discard(beta_filename)

        if args.fir:
            for condition in fir_betas_to_combine:
                beta_frames = np.zeros(shape=(args.fir, activation_betas.shape[1]))
                for f in range(args.fir):
                    beta_column = final_design_unmasked.columns.get_loc(f"{condition}_{f:02d}")
                    beta_frames[f,:] = activation_betas[beta_column,:]
                    beta_img, img_suffix = create_image(
                        data=np.expand_dims(activation_betas[beta_column,:], axis=0),
                        imagetype=imagetype,
                        brain_mask=brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    beta_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}-beta-{condition}-frame-{f}{user_desc}{img_suffix}"
                    descriptions_list.append({
                        "desc_id": f"model-{model_type}-beta-{condition}-frame-0{user_desc}",
                        "description": f"Represents beta weights in a {model_type} model, representing modelled regressor {condition} at frame {f}. {'Additional notes: ' + user_desc if len(user_desc) > 0 else ''}",
                        "model_type": model_type,
                        "condition": condition,
                        "additional_desc": user_desc,
                        "frame": f,
                        "is_nuisance": False
                    })
                    logger.info(f" saving betas for variable {condition} frame {f} to file: {beta_filename}")
                    nib.save(
                        beta_img,
                        beta_filename
                    )
                    unmodified_output_dir_contents.discard(beta_filename)

                beta_img, img_suffix = create_image(
                    data=beta_frames,
                    imagetype=imagetype,
                    brain_mask=brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}-beta-{condition}-concatenated{user_desc}{img_suffix}"
                logger.info(f" saving betas for variable {condition} (all {args.fir} modeled frames) to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )
                unmodified_output_dir_contents.discard(beta_filename)

        # save out the nuisance betas and the residuals
        if flags.debug:
            # save out the nuisance betas
            nuisance_cols = [(i,c) for i,c in enumerate(final_design_unmasked.columns) if len([condition for condition in trial_types if condition in c]) == 0]
            for i, noise_col in nuisance_cols:
                beta_img, img_suffix = create_image(
                    data=np.expand_dims(activation_betas[i,:], axis=0),
                    imagetype=imagetype,
                    brain_mask=brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}-beta-{noise_col}-frame-0{user_desc}{img_suffix}"
                descriptions_list.append({
                    "desc_id": f"model-{model_type}-beta-{noise_col}-frame-0{user_desc}",
                    "description": f"Represents beta weights in a {model_type} model, representing nuisance regressor {noise_col}. {'Additional notes: ' + user_desc if len(user_desc) > 0 else ''}",
                    "model_type": model_type,
                    "condition": noise_col,
                    "additional_desc": user_desc,
                    "frame": 0,
                    "is_nuisance": True
                })
                logger.debug(f" saving betas for nuisance variable: {noise_col} to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )
                unmodified_output_dir_contents.discard(beta_filename)

            # save out residuals of GLM
            resid_img, img_suffix = create_image(
                data=func_residual,
                imagetype=imagetype,
                brain_mask=brain_mask,
                tr=tr,
                header=img_header
            )
            resid_filename = args.output_dir / f"{file_name_base}_desc-model-{model_type}{user_desc}_step-{step_count}_residual{img_suffix}"
            logger.debug(f" saving residual BOLD data after final GLM to file: {resid_filename}")
            nib.save(
                resid_img,
                resid_filename
            )
            unmodified_output_dir_contents.discard(resid_filename)

        # report which files were left unmodified
        if len(unmodified_output_dir_contents) > 0:
            logger.info("the following files were not modified during this run of oceanfla:")
            for unchanged_file in sorted(unmodified_output_dir_contents):
                logger.info(f"\t{unchanged_file.resolve()}")

        log_linebreak()

    except Exception as e:
        logger.exception(e, stack_info=True)
        exit_program_early(str(e))

    descriptions_df = pd.DataFrame(descriptions_list)
    if not descriptions_tsv.is_file():
        descriptions_df.to_csv(descriptions_tsv, sep='\t')
        logger.info(f"Wrote descriptions to {descriptions_tsv.resolve()!s}")
    dataset_description = {
        "Name": f"oceanfla {VERSION}",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative"
    }
    if not dataset_description_json.is_file():
        with dataset_description_json.open("w") as f:
            json.dump(dataset_description, f, indent=4)
            logger.info(f"Wrote dataset description to {dataset_description_json.resolve()!s}")
    logger.info("oceanfla complete!")
    logger.info("\n\n------" + log_stats)


if __name__ == "__main__":
    main()
