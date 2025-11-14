from librosa import ex
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from ..utilities import load_data, create_image_like
from nipype.utils.filemanip import split_filename, fname_presuffix


class RunGLMRegressionInputSpec(BaseInterfaceInputSpec):
    pass


class RunGLMRegressionOutputSpec(TraitedSpec):
    pass


class RunGLMRegression(SimpleInterface):
    input_spec = RunGLMRegressionInputSpec
    output_spec = RunGLMRegressionOutputSpec

    def _run_interface(self, runtime):
        return super()._run_interface(runtime)
    


class ConcatRegressionDataInputSpec(BaseInterfaceInputSpec):
    func_files = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        desc="A list of functional data files"
    )

    event_matrices = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        desc="A list of event matrix files"
    )

    nuisance_matrices = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        None,
        desc="A list of nuisance matrix files"
    )

    tmask_files = event_matrices = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        desc="A list of temporal mask files "
    )

    nuisance_columns = traits.Union(
        traits.List(trait=traits.Str),
        None,
        default_value=None,
        desc="A list of column names to be used from the nuisance matrices."
    )
    
    include_global_mean = traits.Bool(
        default_value=True,
        desc=""
    )

    tasks = traits.Union(
        traits.List(trait=traits.Str),
        traits.Str,
        desc="The task(s) that this regression is for"
    )

    brain_mask = traits.Union( 
        traits.File(exists=True),
        None, 
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )


class ConcatRegressionDataOutputSpec(TraitedSpec):

    func_file_out = traits.File(
        exists=True,
        desc="")
    
    design_file_out = traits.File(
        exists=True,
        desc="")
    
    tmask_file_out = traits.File(
        exists=True,
        desc="")



class ConcatRegressionData(SimpleInterface):
    input_spec = ConcatRegressionDataInputSpec
    output_spec = ConcatRegressionDataOutputSpec

    def _run_interface(self, runtime):
        from ..utilities import replace_entities, load_data, create_image_like
        import numpy as np
        import pandas as pd
        from bids.utils import listify
        from bids.layout import parse_file_entities

        func_files = listify(self.inputs.func_files)
        event_matrices = listify(self.inputs.event_matrices)
        tmask_files = listify(self.inputs.tmask_files)
        nuisance_matrices = listify(self.nuisance_matrices)

        final_func_data, final_design_matrix, final_tmask = combine_regression_data(
            func_data_list= [load_data(f) for f in func_files],
            event_matrices= [pd.read_csv(f) for f in event_matrices],
            tmask_list= [np.loadtxt(f) for f in tmask_files],
            nuisance_matrices= [np.loadtxt(f) for f in nuisance_matrices] if nuisance_matrices else None, 
            nuisance_columns= self.inputs.nuisance_columns,
            global_mean= self.inputs.include_global_mean
        )

        task_label = "-".join(listify(self.inputs.tasks))
        entities_base = {"desc":"glm-input", "task":task_label, "path":None}
        if len(func_files) > 1:
            entities_base["run"] = None

        final_func_file = replace_entities(file=func_files[0], entities=entities_base)
        create_image_like(
            data=final_func_data,
            source_header=func_files[0],
            out_file=final_func_file,
            brain_mask=self.inputs.brain_mask)

        final_design_file = replace_entities(file=event_matrices[0], entities=entities_base.update({"suffix":"design"}))
        final_design_matrix.to_csv(final_design_file, index=False)

        tmask_desc = parse_file_entities(tmask_files[0])["desc"]
        final_tmask_file = replace_entities(file=event_matrices[0], entities=entities_base.update({"desc":f"glm-input-{tmask_desc}"}))
        np.savetxt(final_tmask_file)

        self._results["func_file_out"] = final_func_file
        self._results["design_file_out"] = final_design_file
        self._results["tmask_file_out"] = final_tmask_file
        return runtime
    


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



# def create_final_design(data_list: list[npt.ArrayLike],
#                         design_list: list[tuple[pd.DataFrame, int]],
#                         noise_list: list[pd.DataFrame] = None,
#                         exclude_global_mean: bool = False):
#     """
#     Creates a final, concatenated design matrix for all functional runs in a session

#     Parameters
#     ----------

#     data_list: list[npt.ArrayLike]
#         List of numpy arrays representing BOLD data
#     design_list: list[pd.DataFrame]
#         List of created design matrices corresponding to each respective BOLD run in data_list
#     noise_list: list[pd.DataFrame]
#         List of DataFrame objects corresponding to models of noise for each respective BOLD run in data_list
#     exclude_global_mean: bool
#         Flag to indicate that a global mean should not be included into the final design matrix

#     Returns
#     -------

#     Returns a tuple containing the final concatenated data in index 0, and the
#     final concatenated design matrix in index 1.
#     """
#     num_runs = len(data_list)
#     assert num_runs == len(design_list), "There should be the same number of design matrices and functional runs"

#     design_df_list = [t[0] for t in design_list]
#     if noise_list:
#         assert num_runs == len(noise_list), "There should be the same number of noise matrices and functional runs"
#         for i in range(num_runs):
#             noise_df = noise_list[i]
#             assert len(noise_df) == len(design_df_list[i])
#             run_num = design_list[i][1]
#             rename_dict = dict()
#             for c in noise_df.columns:
#                 if ("trend" in c) or ("mean" in c) or ("spike" in c):
#                     rename_dict[c] = f"run-{run_num:02d}_{c}"
#             noise_df = noise_df.rename(columns=rename_dict)
#             noise_list[i] = noise_df
#             design_df_list[i] = pd.concat([design_df_list[i].reset_index(drop=True), noise_df.reset_index(drop=True)], axis=1)

#     final_design = pd.concat(design_df_list, axis=0, ignore_index=True).fillna(0)
#     if not exclude_global_mean:
#         final_design.loc[:, "global_mean"] = 1
#     final_data = np.concat(data_list, axis=0)
#     return (final_data, final_design)


def combine_regression_data(func_data_list: list,
                            event_matrices: list,
                            tmask_list: list,
                            nuisance_matrices: list = None,
                            nuisance_columns: list[str] = None,
                            global_mean=True):
    import numpy as np
    import pandas as pd

    lengths = [len(x) for x in [func_data_list, event_matrices, tmask_list]]
    if not len(set(lengths)) == 1:
        raise RuntimeError(f"All input lists must be the same length: {set(lengths)}")

    if nuisance_matrices and (len(nuisance_matrices) != lengths[0]):
        raise RuntimeError(f"Expected length of nuisance matrix list to be {lengths[0]} but it was {len(nuisance_matrices)}")
    
    design_list = []
    for i in range(len(event_matrices)):
        event_mat = event_matrices[i]
        time_axis = [len(event_mat), func_data_list[i].shape[0], tmask_list[i].shape[0]]
        if not len(set(time_axis)) == 1:
            raise RuntimeError(f"Grouped functional data, events matrix, and tmask must all have the same number of timepoints, but don't: {set(time_axis)}")
        if nuisance_matrices:
            nuisance_mat = nuisance_matrices[i]
            if len(event_mat) != len(nuisance_mat):
                raise RuntimeError(f"Length of the nuisance matrix ({len(nuisance_mat)}) does not match the length of the data group ({len(event_mat)})")
            if nuisance_columns:
                nuisance_mat = nuisance_mat.loc[:, nuisance_columns]
            event_mat = pd.concat([event_mat.reset_index(drop=True), nuisance_mat.reset_index(drop=True)], axis=1)
        design_list.append(event_mat)

    final_design = pd.concat(design_list, axis=0, ignore_index=True).fillna(0)
    if global_mean:
        final_design.loc[:, "global_mean"] = 1

    final_data = np.concatenate(func_data_list, axis=0)
    final_mask = np.concatenate(tmask_list, axis=0)

    return final_data, final_design, final_mask
