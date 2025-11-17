# from nipype import Node, Workflow, Function, MapNode
# from nipype.pipeline.engine import Node as eNode
# from nipype.interfaces.io import BIDSDataGrabber
import pandas as pd
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
# from . import operations
# from nipype import config as ncfg
from pathlib import Path
# from parser import parse_args
# from config import all_opts
from nipype.utils.filemanip import split_filename, fname_presuffix


class _GenerateNuisanceMatrixInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(
        exists=True,
        desc="Run-specific confounds .csv or .tsv."
    )
    confounds_columns = traits.List(
        desc="Variables to use in nuisance regression before final GLM."
    )
    demean = traits.Bool(
        False
    )
    linear_trend = traits.Bool(
        False,
        desc="Whether or not to include linear trend in the nuisance matrix."
    )
    spike_threshold = traits.Union(
        traits.Str(), None, default_value=None,
        desc="The framewise displacement threshold used when censoring high-motion frames"
    )
    volterra_columns = traits.List(
        desc="Variables to apply volterra expansion to (must be in confound_columns)"
    )
    volterra_lag = traits.Union(
        traits.Int(), None, default_value=None,
        desc="Amount of frames to lag for Volterra expansion."
    )


class _GenerateNuisanceMatrixOutputSpec(TraitedSpec):
    nuisance_matrix = OutputMultiObject(
        File(exists=True),
        desc="Outputted nuisance matrix as a file."
    )


class GenerateNuisanceMatrix(SimpleInterface):
    """
    Generates a nuisance matrix for regression before final GLM.
    """

    input_spec = _GenerateNuisanceMatrixInputSpec
    output_spec = _GenerateNuisanceMatrixOutputSpec

    def _run_interface(self, runtime):
        from ..utilities import replace_entities

        out_file = replace_entities(self.inputs.confounds_file, 
                                    {"desc":"nuisance", "suffix":"matrix", "ext":".csv", "path":runtime.cwd})
        generate_nuisance_matrix(
            confounds_file=self.inputs.confounds_file,
            confounds_columns=self.inputs.confounds_columns,
            output_path=out_file,
            demean=self.inputs.demean,
            linear_trend=self.inputs.linear_trend,
            spike_threshold=self.inputs.spike_threshold,
            volterra_lag=self.inputs.volterra_lag,
            volterra_columns=self.inputs.volterra_columns
        )
        self._results["nuisance_matrix"] = out_file
        return runtime

    # def _list_outputs(self):
    #     return {'nuisance_matrix': self.inputs.out_file}


def generate_nuisance_matrix(confounds_file: str,
                             confounds_columns: list,
                             output_path: str | Path,
                             demean: bool = False,
                             linear_trend: bool = False,
                             spike_threshold: float = None,
                             volterra_lag: int = None,
                             volterra_columns: list = None,):
    
    confounds_columns = set(confounds_columns)
    if spike_threshold:
        confounds_columns.add("framewise_displacement")
    if volterra_columns:
        confounds_columns.update(volterra_columns)
    suffix = "." + confounds_file.split(".")[-1]
    if suffix == ".csv":
        nuisance = pd.read_csv(confounds_file).loc[:,list(confounds_columns)]
    elif suffix == ".tsv":
        nuisance = pd.read_csv(confounds_file, sep='\t').loc[:,list(confounds_columns)]
    else:
        raise ValueError("Invalid suffix (must be .csv or .tsv)")
    if "framewise_displacement" in confounds_columns:
        nuisance.loc[0, "framewise_displacement"] = 0
        if spike_threshold:
            b = 0
            for a in range(len(nuisance)):
                if nuisance.loc[a, "framewise_displacement"] > spike_threshold:
                    spike_col = make_regressor_run_specific(confounds_file, f"spike{b}")
                    nuisance[spike_col] = 0
                    nuisance.loc[a, spike_col] = 1
                    b += 1
    if demean:
        nuisance[make_regressor_run_specific(confounds_file, "mean")] = 1
    if linear_trend:
        nuisance[make_regressor_run_specific(confounds_file, "trend")] = np.arange(0, len(nuisance))
    if volterra_columns and volterra_lag:
        for vc in volterra_columns:
            for lag in range(volterra_lag):
                nuisance.loc[:, f"{vc}_{lag + 1}"] = nuisance.loc[:, vc].shift(lag + 1)
        nuisance.fillna(0, inplace=True)

    if str(output_path).endswith(".tsv"):
        nuisance.to_csv(output_path, sep='\t', index=False)
        return output_path
    elif str(output_path).endswith(".csv"):
        nuisance.to_csv(output_path, index=False)
        return 
    


def make_regressor_run_specific(regressor_name:str, bids_source_file:str|Path=None,  run=None, task=None):
    if (run is None) or (task is None):
        if bids_source_file is None: 
            raise RuntimeError("'run' and 'task' parameters must be provided if 'bids_source_file' is not provided.")
        from ..config import get_bids_file
        bidsfile = get_bids_file(bids_source_file)
        run = int(bidsfile.entities.get("run", 1))
        task = bidsfile.entities["task"]
    return f"task-{task}-run-{run}-{regressor_name}"