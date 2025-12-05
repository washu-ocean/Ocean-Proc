from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)


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

        beta_files, beta_labels, func_residual_file = massuni_linGLM(
            func_file=self.inputs.bold_file_in,
            design_matrix=self.inputs.design_matrix,
            tmask_file=self.inputs.tmask_file,
            stdscale=self.inputs.stdscale,
            brain_mask=self.inputs.brain_mask
        )

        self._results["beta_files"] = beta_files
        self._results["beta_labels"] = beta_labels
        self._results["residual_bold_file"] = func_residual_file

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

    regressor_columns = traits.Union(
        traits.List(trait=traits.Str),
        None,
        default_value=None,
        desc="A list of column names to be used from the nuisance matrices."
    )

    inclusion_list = traits.Union(
        traits.List(trait=traits.Bool),
        None,
        default_value=None,
        desc="A list of boolean values to indicate inclusion in the final concatenated data"
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

    tmask_file = traits.Union(
        traits.File(exists=True),
        None,
        desc="The concatenation of the input list 'tmask_files_in'"
    )

    residual_design_matrix = traits.Union(
        traits.File(exists=True),
        None,
        desc="The design matrix using the columns that are not needed for the current regression"
    )


class ConcatRegressionData(SimpleInterface):
    input_spec = ConcatRegressionDataInputSpec
    output_spec = ConcatRegressionDataOutputSpec

    def _run_interface(self, runtime):
        from bids.utils import listify

        func_files = listify(self.inputs.bold_files_in)
        event_matrices = listify(self.inputs.event_matrices)
        tmask_files = listify(self.inputs.tmask_files_in)
        nuisance_matrices = listify(self.inputs.nuisance_matrices)

        final_func_file, final_tmask, final_design_matrix, residual_design_matrix = combine_regression_data(
            func_list=func_files,
            event_matrix_files=event_matrices,
            tmask_files=tmask_files,
            nuisance_matrix_files=nuisance_matrices,
            regressor_columns=self.inputs.regressor_columns,
            inclusion_list=self.inputs.inclusion_list,
            global_mean=self.inputs.include_global_mean,
            tasks=self.inputs.tasks,
            brain_mask=self.inputs.brain_mask
        )

        self._results["bold_file"] = final_func_file
        self._results["tmask_file"] = final_tmask
        self._results["design_matrix"] = final_design_matrix
        self._results["residual_design_matrix"] = residual_design_matrix

        return runtime


def massuni_linGLM(func_file: str,
                   design_matrix: str,
                   tmask_file: str,
                   stdscale: bool,
                   brain_mask: str = None):
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
    from ..utilities import load_data, create_image_like, replace_entities
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    func_data = load_data(func_file, brain_mask)
    design_matrix = pd.read_csv(design_matrix, sep="\t")
    design_matrix_data = design_matrix.to_numpy()
    mask = np.loadtxt(tmask_file).astype(bool) if tmask_file else np.full(
        shape=(func_data.shape[0],), fill_value=True)

    assert design_matrix_data.shape[0] == func_data.shape[
        0], "the design matrix must be the same length as the functional data"
    assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"

    # apply the mask to the data
    masked_func_data = func_data.copy()[mask, :]
    masked_design_matrix = design_matrix_data.copy()[mask, :]

    # standardize the masked data
    func_ss = StandardScaler()
    design_ss = StandardScaler()
    if stdscale:
        masked_func_data = func_ss.fit_transform(masked_func_data)
        masked_design_matrix = design_ss.fit_transform(masked_design_matrix)

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

    # save the data out
    entities_base = {"desc": "modelOutput", "path": None}

    # beta files
    beta_files, beta_labels, = [], []
    for i, beta_label in enumerate(design_matrix.columns):
        beta_filename = replace_entities(
            file=func_file,
            entities=entities_base | {"suffix": f"beta-{beta_label}_boldmap"}
        )
        create_image_like(
            data=(beta_data[i])[np.newaxis, :],
            source_header=func_file,
            out_file=beta_filename,
            dscalar_axis=[f"beta-{beta_label}"],
            brain_mask=brain_mask
        )
        beta_files.append(beta_filename)
        beta_labels.append(beta_label)

    # residual bold data
    residual_filename = replace_entities(
        file=func_file,
        entities=entities_base | {"suffix": "residual-bold"}
    )
    create_image_like(
        data=resids,
        source_header=func_file,
        out_file=residual_filename,
        brain_mask=brain_mask
    )
    return (beta_files, beta_labels, residual_filename)


def combine_regression_data(func_list: list,
                            tasks: list,
                            tmask_files: list = None,
                            event_matrix_files: list = None,
                            nuisance_matrix_files: list = None,
                            regressor_columns: list[str] = None,
                            inclusion_list: list[bool] = None,
                            global_mean=True,
                            brain_mask: str = None):
    from ..utilities import replace_entities, load_data, create_image_like
    import numpy as np
    import pandas as pd

    func_data_list = [load_data(f, brain_mask) for f in func_list]
    event_matrices = [pd.read_csv(
        f, sep="\t") for f in event_matrix_files] if event_matrix_files else None
    tmask_list = [np.loadtxt(f) for f in tmask_files] if tmask_files else None
    nuisance_matrices = [pd.read_csv(
        f, sep="\t") for f in nuisance_matrix_files] if nuisance_matrix_files else None

    lengths = [len(x) for x in
               [func_data_list, tmask_list, event_matrices,
                   nuisance_matrices, inclusion_list]
               if x is not None]
    if not len(set(lengths)) == 1:
        raise RuntimeError(
            f"All input lists must be the same length: {lengths}")
    needed_len = lengths[0]

    # remove any runs that are being excluded
    if inclusion_list:
        remaining_data_lists = [[x for i, x in enumerate(data_list) if inclusion_list[i]]
                                if data_list else None for data_list in
                                [func_data_list, tmask_list, event_matrices, nuisance_matrices]]
        func_data_list, tmask_list, event_matrices, nuisance_matrices = remaining_data_lists

    # need either event matrices or nuisance matrices
    input_lists = [l for l in [event_matrices,
                               nuisance_matrices] if l is not None]
    if len(input_lists) < 1:
        raise RuntimeError(
            f"Regression data must include event data or nuisance data, but recieved neither")

    # combine the data matrices if needed
    design_data_list = [tuple(in_list[i] for in_list in input_lists)
                        for i in range(needed_len)]
    for i in range(needed_len):
        time_axis = [len(design_data_list[i][0]), func_data_list[i].shape[0]]
        if len(design_data_list[i]) == 2:
            time_axis.append(len(design_data_list[i][1]))
        if not len(set(time_axis)) == 1:
            raise RuntimeError(
                f"Grouped functional data and events matrix must all have the same number of timepoints, but don't: {set(time_axis)}")

        run_design = (
            pd.concat([design_data_list[i][0].reset_index(drop=True),
                       design_data_list[i][1].reset_index(drop=True)], axis=1)
        ) if len(design_data_list[i]) == 2 else (
            design_data_list[i][0])

        design_data_list[i] = run_design

    # concatenate all of the data on the time axis
    task_label = "-".join(tasks)
    entities_base = {"desc": "modelInput", "task": task_label, "path": None}
    if len(func_list) > 1:
        entities_base["run"] = None
    res_list = []

    # functional data
    final_func_file = replace_entities(
        file=func_list[0],
        entities=entities_base
    )
    create_image_like(
        data=np.concatenate(func_data_list, axis=0),
        source_header=func_list[0],
        out_file=final_func_file,
        brain_mask=brain_mask)
    res_list.append(final_func_file)

    # tmask data
    final_tmask = None
    if tmask_list is not None:
        for i, run_tm in enumerate(tmask_list):
            if run_tm.shape[0] != func_data_list[i].shape[0]:
                raise RuntimeError(
                    f"Grouped functional data and tmasks must all have the same number of timepoints, but don't: {func_data_list[i].shape[0]}, {run_tm.shape[0]}")
        final_tmask = replace_entities(
            file=tmask_files[0], entities=entities_base)
        np.savetxt(final_tmask,
                   np.concatenate(tmask_list, axis=0))
    res_list.append(final_tmask)

    # design data
    final_design = pd.concat(design_data_list, axis=0,
                             ignore_index=True).fillna(0)
    residual_design_file = None
    if regressor_columns:
        design_columns = final_design.columns.to_list()
        residual_columns = [
            dc for dc in design_columns if dc not in regressor_columns]
        residual_design = final_design.loc[:, residual_columns]
        final_design = final_design.loc[:, regressor_columns]

        residual_design_file = replace_entities(
            file=event_matrix_files[0],
            entities=entities_base | {"suffix": "residual-design"}
        )
        residual_design.to_csv(residual_design_file, index=False, sep="\t")

    if global_mean:
        final_design.loc[:, "global_mean"] = 1

    final_design_file = replace_entities(
        file=event_matrix_files[0],
        entities=entities_base | {"suffix": "design"}
    )
    final_design.to_csv(final_design_file, index=False, sep="\t")

    res_list.append(final_design_file)
    res_list.append(residual_design_file)

    return res_list
