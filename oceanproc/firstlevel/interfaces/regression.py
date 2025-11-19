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
    bold_file_in = traits.File(
        exists=True,
        desc="The functional data file"
    )

    design_matrix = traits.File(
        exists=True,
        desc="The design matrix for the regression"
    )

    tmask_file = traits.Union(
        traits.File(exists=True),
        None,
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
    
    residual_bold_file = traits.File(
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
            func_data=load_data(self.inputs.bold_file_in),
            design_matrix=pd.read_csv(self.inputs.design_matrix, sep="\t"),
            mask=np.loadtxt(self.inputs.tmask_file) if self.inputs.tmask_file else None,
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
        self._results["residual_bold_file"] = residual_filename
        return runtime


class ConcatRegressionDataInputSpec(BaseInterfaceInputSpec):
    bold_files_in = traits.Union(
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

    tmask_files_in = event_matrices = traits.Union(
        traits.List(trait=traits.File(exists=True)),
        traits.File(exists=True),
        desc="A list of temporal mask files "
    )

    include_in_glm = traits.Union(
        traits.List(trait=traits.Bool),
        traits.Bool,
        desc="List of Bool that denotes whether a run should be included in the concatenated GLM or not."
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

    bold_file = traits.File(
        exists=True,
        desc="")
    
    design_matrix = traits.File(
        exists=True,
        desc="")
    
    tmask_file = traits.File(
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

        func_files = listify(self.inputs.bold_files_in)
        event_matrices = listify(self.inputs.event_matrices)
        tmask_files = listify(self.inputs.tmask_files_in)
        nuisance_matrices = listify(self.nuisance_matrices)

        idx = 0
        for include in listify(self.inputs.include_in_glm):
            if not include:
                func_files.pop(idx)
                event_matrices.pop(idx)
                tmask_files.pop(idx)
                nuisance_matrices.pop(idx)
            else:
                idx += 1
        del idx

        if len(func_files) == 0:
            raise ValueError("No BOLD runs remain after filtering by exclusion criteria.")
            

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

        self._results["bold_file"] = final_func_file
        self._results["design_matrix"] = final_design_file
        self._results["tmask_file"] = final_tmask_file

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
    if mask:
        assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"
        assert mask.dtype == bool
    else:
        mask = np.full(shape=(func_data.shape[0],), fill_value=True)

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
            
