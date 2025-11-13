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

