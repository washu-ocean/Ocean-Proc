from numba import jit
import torch
import nibabel as nib
import numpy as np
import scipy
from .utils import set_device

# @jit(nopython=True)
# def ols(y, x):
    # if len(y.shape) != 2:
        # raise ValueError(f"Shape of random variable vector y must be of length 2. y.shape: {y.shape}")
    # elif len(x.shape) != 2:
        # raise ValueError(f"Shape of design matrix x must be of length 2. x.shape: {x.shape}")
    # elif y.shape[0] != x.shape[0]:
        # raise ValueError(f"The number of rows in y and x must match.\ny rows: {y.shape[0]}\nx rows: {x.shape[0]}")
    # beta_hat = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
    # residuals = y - (x @ beta_hat)
    # return (beta_hat, residuals)

def ols_regression_over_dscalars(combined_dscalars: nib.cifti2.cifti2.Cifti2Image,
                                 design_mat: np.ndarray|torch.Tensor):
    y = torch.from_numpy(combined_dscalars.get_fdata())
    if isinstance(design_mat, np.ndarray):
        x = torch.from_numpy(design_mat)
    if not torch.is_tensor(x):
        raise TypeError("design_mat must be a torch tensor or numpy ndarray")
    device = set_device()
    y, x = y.to(device), x.to(device)
    beta_hat = torch.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
    residuals = y - (x @ beta_hat)
    return (beta_hat, residuals)

# @jit(nopython=True)
# def ttest_1samp_over_dscalars(combined_dscalars: nib.cifti2.cifti2.Cifti2Image,
                              # output_path: str,
                              # popmean: float = 0,
                              # alpha: float = 0.05,
                              # tails: int = 1):
    # assert tails in (1,2), "tails must be either 1 or 2"
    # assert alpha > 0 and alpha < 1, "alpha must be between 0 and 1"
    # n = combined_dscalars.shape[0]
    # df = n - 1
    # sample_means = np.mean(combined_dscalars, axis=0)
    # sample_error = (np.std(combined_dscalars, axis=0)) / np.sqrt(n)
    # diff_mean = sample_means - popmean
    # t_scores = diff_mean / sample_error
    # p_values = tails * (1 - scipy.stats.t.cdf(abs(t_scores), df))
    # p_values = np.array([p_values])
    # header = nib.cifti2.cifti2.Cifti2Header(p_values)
    # img = nib.cifti2.cifti2.Cifti2Image(dataobj=p_values, header=header)
    # nib.save(img, output_path)

    


     
