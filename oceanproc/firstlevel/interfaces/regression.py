from mimetypes import suffix_map
from librosa import ex
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from oceanproc.firstlevel.interfaces import nuisance
from ..utilities import load_data, create_image_like
from nipype.utils.filemanip import split_filename, fname_presuffix


class RunGLMRegressionInputSpec(BaseInterfaceInputSpec):
    func_file = traits.File(
        exists=True,
        desc="The functional data file"
    )

    design_matrix = traits.File(
        exists=True,
        desc="The design matrix for the regression"
    )

    tmask_file = traits.File(
        exists=True,
        desc="The temporal mask file",
    )

    stdscale = traits.Bool(
        default_value=False,
        desc="Flag to indicate standard scaling the data before the regression"
    )

    brain_mask = traits.Union( 
        traits.File(exists=True),
        None, 
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )


class RunGLMRegressionOutputSpec(TraitedSpec):
    beta_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )
    
    beta_labels = traits.List(
        trait=traits.Str,
        desc=""
    )
    
    func_residual_file = traits.File(
        exists=True,
        desc="")


class RunGLMRegression(SimpleInterface):
    input_spec = RunGLMRegressionInputSpec
    output_spec = RunGLMRegressionOutputSpec

    def _run_interface(self, runtime):
        from ..utilities import load_data, create_image_like, replace_entities
        import pandas as pd
        import numpy as np
        beta_map, func_residuals = massuni_linGLM(
            func_data=load_data(self.inputs.func_file),
            design_matrix=pd.read_csv(self.inputs.design_matrix, sep="\t"),
            mask=np.loadtxt(self.inputs.tmask_file),
            stdscale=self.inputs.stdscale
        )

        entities_base = {"desc":"modelOutput", "path":runtime.cwd}
        beta_files, beta_labels, = [], []
        for beta_label, beta_data in beta_map.items():
            beta_filename = replace_entities(
                file=self.inputs.func_file,
                entities=entities_base.update({"desc":f"beta-{beta_label}", "suffix":"boldmap"})
            )
            create_image_like(
                data=beta_data[np.newaxis,:],
                source_header=self.inputs.func_file,
                out_file=beta_filename,
                dscalar_axis=[f"beta-{beta_label}"],
                brain_mask=self.inputs.brain_mask
            )
            beta_files.append(beta_filename)
            beta_labels.append(beta_label)
        
        residual_filename = replace_entities(
            file=self.inputs.func_file,
            entities=entities_base.update({"desc":"residual"})
        )
        create_image_like(
            data=func_residuals,
            source_header=self.inputs.func_file,
            out_file=residual_filename,
            brain_mask=self.inputs.brain_mask
        )

        self._results["beta_files"] = beta_files
        self._results["beta_labels"] = beta_labels
        self._results["func_residual_file"] = residual_filename
        return runtime


class ConcatRegressionDataInputSpec(BaseInterfaceInputSpec):
    func_files = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        desc="A list of functional data files"
    )

    event_matrices = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        None,
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

    regressor_columns = traits.Union(
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
    
    residual_design_file = traits.Union(
        traits.File(exists=True),
        None,
        desc="The design matrix using the columns that are not needed for the current regression"
    )



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

        final_func_data, final_tmask, final_design_matrix, residual_design_matrix = combine_regression_data(
            func_data_list= [load_data(f) for f in func_files],
            event_matrices= [pd.read_csv(f) for f in event_matrices] if nuisance_matrices else None,
            tmask_list= [np.loadtxt(f) for f in tmask_files],
            nuisance_matrices= [np.loadtxt(f) for f in nuisance_matrices] if nuisance_matrices else None, 
            regressor_columns= self.inputs.regressor_columns,
            global_mean= self.inputs.include_global_mean
        )

        task_label = "-".join(listify(self.inputs.tasks))
        entities_base = {"desc":"modelInput", "task":task_label, "path":runtime.cwd}
        if len(func_files) > 1:
            entities_base["run"] = None

        final_func_file = replace_entities(file=func_files[0], entities=entities_base)
        create_image_like(
            data=final_func_data,
            source_header=func_files[0],
            out_file=final_func_file,
            brain_mask=self.inputs.brain_mask)
        
        tmask_desc = parse_file_entities(tmask_files[0])["desc"]
        final_tmask_file = replace_entities(file=event_matrices[0], entities=entities_base.update({"desc":f"modelInput-{tmask_desc}"}))
        np.savetxt(final_tmask_file, final_tmask)

        final_design_file = replace_entities(file=event_matrices[0], entities=entities_base.update({"suffix":"design"}))
        final_design_matrix.to_csv(final_design_file, index=False, sep="\t")

        self._results["func_file_out"] = final_func_file
        self._results["design_file_out"] = final_design_file
        self._results["tmask_file_out"] = final_tmask_file

        if residual_design_matrix:
            residual_design_file = replace_entities(file=event_matrices[0], entities=entities_base.update({"suffix":"design", "desc":"unused"}))
            residual_design_matrix.to_csv(residual_design_file, index=False, sep="\t")
            self._results["residual_design_file"] = residual_design_file
        else:
            self._results["residual_design_file"] = None

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
    design_matrix_data = design_matrix.to_numpy()

    func_ss = StandardScaler()
    design_ss = StandardScaler()
    if stdscale:
        masked_func_data = func_ss.fit_transform(func_data.copy()[mask, :])
        masked_design_matrix = design_ss.fit_transform(design_matrix_data.copy()[mask, :])
    else:
        masked_func_data = func_data.copy()[mask, :]
        masked_design_matrix = design_matrix_data.copy()[mask, :]

    # standardize the masked data

    # comput beta values
    inv_mat = np.linalg.pinv(masked_design_matrix)
    beta_data = np.dot(inv_mat, masked_func_data)

    # standardize the unmasked data
    if stdscale:
        func_data = func_ss.transform(func_data)
        design_matrix_data = design_ss.transform(design_matrix_data)

    # compute the residuals with unmasked data
    est_values = np.dot(design_matrix_data, beta_data)
    resids = func_data - est_values

    beta_map = {c: beta_data[i] for i, c in enumerate(design_matrix.columns)}
    return (beta_map, resids)



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
                            tmask_list: list,
                            event_matrices: list = None,
                            nuisance_matrices: list = None,
                            regressor_columns: list[str] = None,
                            global_mean=True):
    import numpy as np
    import pandas as pd

    lengths = [len(x) for x in [func_data_list, tmask_list]]
    if not len(set(lengths)) == 1:
        raise RuntimeError(f"All input lists must be the same length: {set(lengths)}")
    needed_len = lengths[0]
    
    # need either event matrices or nuisance matrices
    input_lists = []
    for l in [event_matrices, nuisance_matrices]:
        if l:
            if len(l) != needed_len:
                raise RuntimeError(f"Expected length of input list to be {needed_len} but it was {len(l)}")
            input_lists.append(l)
    if len(input_lists) < 1:
        raise RuntimeError(f"Regression data must include event data or nuisance data, but recieved neither")
    
    # combine the data matrices if needed
    design_data_list = [tuple([in_list[i] for in_list in input_lists]) for i in range(needed_len)]
    for i in range(needed_len):
        time_axis = [len(design_data_list[i][0]), func_data_list[i].shape[0], tmask_list[i].shape[0]]
        if len(design_data_list[i]) == 2:
            time_axis.append(len(design_data_list[i][1]))
        if not len(set(time_axis)) == 1:
            raise RuntimeError(f"Grouped functional data, events matrix, and tmask must all have the same number of timepoints, but don't: {set(time_axis)}")
        
        run_design = ( 
            pd.concat([design_data_list[i][0].reset_index(drop=True), 
                    design_data_list[i][1].reset_index(drop=True)], axis=1)
                    ) if len(design_data_list[i]) == 2 else ( 
            design_data_list[i][0])
        
        # only grab the requested columns if needed
        # if regressor_columns:
        #     run_design = run_design.loc[:, regressor_columns]
        
        design_data_list[i] = run_design
    
    # concatenate all of the data on the time axis
    res_list = [
        np.concatenate(func_data_list, axis=0), # concat the func data
        np.concatenate(tmask_list, axis=0), # concat the tmask data
    ]

    final_design = pd.concat(design_data_list, axis=0, ignore_index=True).fillna(0)
    residual_design = None
    if regressor_columns:
        design_columns = final_design.columns.to_list()
        residual_columns = [dc for dc in design_columns if dc not in regressor_columns]
        residual_design = final_design.loc[:, residual_columns]
        final_design = final_design.loc[:, regressor_columns]

    if global_mean:
        final_design.loc[:, "global_mean"] = 1
    res_list.append(final_design)
    res_list.append(residual_design)

    return res_list
            
    # design_list = []
    # for i in range(len(event_matrices)):
    #     event_mat = event_matrices[i]
    #     time_axis = [len(event_mat), func_data_list[i].shape[0], tmask_list[i].shape[0]]
    #     if not len(set(time_axis)) == 1:
    #         raise RuntimeError(f"Grouped functional data, events matrix, and tmask must all have the same number of timepoints, but don't: {set(time_axis)}")
    #     if nuisance_matrices:
    #         nuisance_mat = nuisance_matrices[i]
    #         if len(event_mat) != len(nuisance_mat):
    #             raise RuntimeError(f"Length of the nuisance matrix ({len(nuisance_mat)}) does not match the length of the data group ({len(event_mat)})")
    #         if regressor_columns:
    #             nuisance_mat = nuisance_mat.loc[:, regressor_columns]
    #         event_mat = pd.concat([event_mat.reset_index(drop=True), nuisance_mat.reset_index(drop=True)], axis=1)
    #     design_list.append(event_mat)

    # final_design = pd.concat(design_list, axis=0, ignore_index=True).fillna(0)
    # if global_mean:
    #     final_design.loc[:, "global_mean"] = 1

    # final_data = np.concatenate(func_data_list, axis=0)
    # final_mask = np.concatenate(tmask_list, axis=0)

    # return final_data, final_design, final_mask
