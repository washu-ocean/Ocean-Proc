from nipype.interfaces.base import SimpleInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec
from pathlib import Path
import pandas as pd
import numpy as np
from nipype import logging, Function
from textwrap import dedent
from nipype.utils.filemanip import split_filename, fname_presuffix
from scipy.stats import gamma


# events_to_design
logger = logging.getLogger("nipype.interface")


# class ExtractRunFilesInputSpec(BaseInterfaceInputSpec):
#     run_files = traits.Dict(key_trait=traits.Str, has_items=True)

# class ExtractRunFilesOutputSpec(TraitedSpec)



class EventsMatrixInputSpec(BaseInterfaceInputSpec):
    event_file = File(exists=True, mandatory=True, 
                        desc="A BIDS style events file of type .tsv")
    
    # bold_file = File(exists=True,)
    
    fir = traits.Union(
        traits.Int(desc="An integer denoting the order of an FIR filter", mandatory=False, allow_none=True),
        None,
        default_value=None)
    
    hrf = traits.Union(
        traits.List(trait=traits.Int, minlen=2, maxlen=2,
                        desc="A 2-element list, where hrf[0] denotes the time to the peak of an HRF, and hrf[1] denotes the duration of its 'undershoot' after the peak."), 
        traits.File(exists=True),
        None, 
        default_value=None)
    
    fir_vars = traits.Union(
        traits.List(trait=traits.Str, 
                        desc="A list of column names denoting which columns should have an FIR filter applied."),
        None,
        default_value=None)
    
    hrf_vars = traits.Union(
        traits.List(trait=traits.Str, 
                        desc="A list of column names denoting which columns should be convolved with the HRF function defined in the hrf list."),
        None,
        default_value=None)
    
    unmodeled = traits.Union(
        traits.List(trait=traits.Str, 
                        desc="A list of column names denoting which columns should not be modeled by neither hrf or fir, but still included in the design matrix."),
        None,
        default_value=None)
    
    volumes = traits.Int(desc="The number of volumes that are in the corresponding BOLD run")

    tr = traits.Float(desc="The Repetition Time for this BOLD run")
    

class EventsMatrixOutputSpec(TraitedSpec):
    events_matrix = File(exists=True, 
                    desc="A run-level design matrix created using the input parameters")
    

class EventsMatrix(SimpleInterface):
    input_spec = EventsMatrixInputSpec
    output_spec = EventsMatrixOutputSpec
    
    def _run_interface(self, runtime):
        from .utilities import replace_entities

        out_file = replace_entities(self.inputs.event_file, 
                                    {"desc": "events", "suffix":"matrix", "ext":".csv", "path":runtime.cwd})
        make_design_matrix(
            event_file=self.inputs.event_file,
            tr=self.inputs.tr,
            volumes=self.inputs.volumes,
            fir=self.inputs.fir,
            fir_vars=self.inputs.fir_vars,
            hrf=self.inputs.hrf,
            hrf_vars=self.inputs.hrf_vars,
            unmodeled=self.inputs.unmodeled,
            output_path=out_file
        )
        self._results["events_matrix"] = out_file
        return runtime
    
    # def _list_outputs(self):
    #     outputs = self._outputs().get()
    #     event_file = self.inputs.event_file
    #     outputs["out_file"] = Path(split_filename(event_file)[1] + "_events-matrix.csv").resolve()
    #     return outputs
        

def make_design_matrix(event_file: str|Path,
                       volumes:int,
                       tr: float,
                       fir: int = None,
                       hrf: list[int] | Path = None,
                       fir_vars: list[str] = None,
                       hrf_vars: list[str] = None,
                       unmodeled: list[str] = None,
                       output_path: str|Path = None):
    events_long = _make_events_long(event_file, volumes, tr)
    events_matrix = events_long.copy()

    # If both FIR and HRF are specified, we should have at least one list
    # of columns for one of the categories specified.
    if (fir and hrf) and not (fir_vars or hrf_vars):
        raise RuntimeError("Both FIR and HRF were specified, but you need to specify at least one list of variables (fir_vars or hrf_vars)")
    # fir_vars and hrf_vars must not have overlapping columns
    if (fir_vars and hrf_vars) and not set(fir_vars).isdisjoint(hrf_vars):
        raise RuntimeError("Both FIR and HRF lists of variables were specified, but they overlap.")
    conditions = [s for s in np.unique(events_matrix.columns)]  # unique trial types
    residual_conditions = [c for c in conditions if c not in unmodeled] if unmodeled else conditions
    if (fir and hrf) and (bool(fir_vars) ^ bool(hrf_vars)):  # Create other list if only one is specified
        if fir_vars:
            hrf_vars = [c for c in residual_conditions if c not in fir_vars]
        elif hrf_vars:
            fir_vars = [c for c in residual_conditions if c not in hrf_vars]
        assert set(hrf_vars).isdisjoint(fir_vars)

    if fir:
        fir_conditions = residual_conditions
        if fir_vars and len(fir_vars) > 0:
            fir_conditions = [c for c in residual_conditions if c in fir_vars]
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
        if hrf_vars and len(hrf_vars) > 0:
            hrf_conditions = [c for c in residual_conditions if c in hrf_vars]
        residual_conditions = [c for c in residual_conditions if c not in hrf_conditions]
        logger.info(events_matrix)
        cfeats = _hrf_convolve_features(features=events_matrix,
                                    column_names=hrf_conditions,
                                    time_col = 'index',
                                    units= 's',
                                    time_to_peak=(hrf[0] if isinstance(hrf, list) else None),
                                    undershoot_dur=(hrf[1] if isinstance(hrf, list) else None),
                                    custom_hrf=(hrf if isinstance(hrf, Path) else None))
        for c in hrf_conditions:
            events_matrix[c] = cfeats[c]

    if len(residual_conditions) > 0:
        logger.warning(dedent(f"""The following trial types were not selected under either of the specified models
                        and were also not selected to be left unmodeled. These variables will not be included in the design matrix:\n\t {residual_conditions}"""))
        events_matrix = events_matrix.drop(columns=residual_conditions)
    
    if output_path:
        events_matrix.to_csv(output_path, index=False)



def _make_events_long(event_file:Path, volumes: int, tr:float):
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

    duration = tr * volumes
    events_df = pd.read_csv(event_file, index_col=None, delimiter="\t")
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(0, columns=conditions, index=np.arange(0,duration,tr)[:volumes])

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e,'onset'])
        events_long.loc[i, events_df.loc[e,'trial_type']] = 1
        if events_df.loc[e,'duration'] > tr:
            offset = events_df.loc[e,'onset'] + events_df.loc[e,'duration']
            j = find_nearest(events_long.index, offset)
            events_long.loc[i:j, events_df.loc[e,'trial_type']] = 1

    # if output_file and output_file.suffix == ".csv":
    #     logger.debug(f" saving events long to file: {output_file}")
    #     events_long.to_csv(output_file)

    return events_long
    
def _hrf_convolve_features(features: pd.DataFrame = None,
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
    hrf_sig = np.loadtxt(custom_hrf) if custom_hrf is not None else self._create_hrf(time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)

    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features
    

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
    


def _extract_task_run_group(bold_list:list, confounds_list:list, events_list:list, task_needed, run_needed):
    from oceanproc.firstlevel.config import get_bids_file
    run_dict = {
        "bold":None,
        "confounds":None,
        "events":None
    }
    for ftype, file_list in {"bold":bold_list, "confounds":confounds_list, "events":events_list}.items():
        for file in file_list:
            bids_file = get_bids_file(file)
            run = int(bids_file.entities.get("run", 1))
            if run == int(run_needed) and bids_file.entities["task"] == task_needed:
                run_dict[ftype] = bids_file
                break
    
    if not all(list(run_dict.values())):
        raise RuntimeError(f"Could not find all the needed files for run-{run_needed}")
    
    return run_dict["bold"], run_dict["confounds"], run_dict["events"]

ExtractTaskRunGroup = Function(
    function=_extract_task_run_group,
    input_names=["bold_list", "confounds_list", "events_list", "task_needed", "run_needed"],
    output_names=["bold_file",
                "confounds_file",
                "events_file"]
)


def get_number_of_volumes(bold_in, brain_mask = None):
    from oceanproc.firstlevel.utilities import load_data
    func_data = load_data(func_file=bold_in, 
                                    brain_mask=brain_mask)
    return func_data.shape[0]


GetVolumeCount = Function(
    function=get_number_of_volumes,
    input_names=["bold_in", "brain_mask"],
    output_names=["volumes"]
)

# def group_runs(bolds:list, confounds:list, events:list):
#     from oceanproc.firstlevel.config import get_layout_for_file
#     run_dict = dict()
#     for ftype, file_list in {"bold":bolds, "confounds":confounds, "events":events}.items():
#         for file in file_list:
#             layout = get_layout_for_file(file)
#             bfile = layout.get_file(file)
#             run = int(bfile.entities["run"]) if "run" in bfile.entities else 1
#             if run in run_dict:
#                 run_dict[run][ftype] = bfile
#             else:
#                 run_dict[run] = {ftype: bfile}
            
#     return [v for k,v in sorted(run_dict.items(), key=lambda x: x[1])]
        

# def extract_run_files(run_dict:dict):
#     return 


