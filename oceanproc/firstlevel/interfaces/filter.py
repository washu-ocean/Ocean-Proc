from pathlib import Path
from copy import deepcopy
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)
import numpy as np
import numpy.typing as npt
import nibabel as nib
from nilearn.signal import (
    butterworth,
    _handle_scrubbed_volumes
)


def filter_data(func_data: npt.ArrayLike,
                mask: npt.ArrayLike,
                t_r: float,
                low_pass: float = 0.1,
                high_pass: float = 0.008,
                padtype: str = "none",
                padlen: int = 50):
    if not any((
        padtype == "none",
        padlen is None,
        (padtype != "zero" and padlen is not None and padlen > 0),
        (padtype == "zero" and padlen is not None and padlen >= 2),
    )):
        raise ValueError(f"Pad length of {padlen} incompatible with pad type {'odd' if padtype is None else padtype}")

    padded_func_data = np.pad(func_data, ((padlen, padlen), (0, 0)), mode='constant', constant_values=0)
    padded_mask = np.pad(mask, (padlen, padlen), mode='constant', constant_values=True)

    # if the mask is excluding frames, interpolate the censored frames
    if np.sum(mask) < mask.shape[0]:
        padded_func_data, _, padded_mask = _handle_scrubbed_volumes(
            signals=padded_func_data,
            confounds=None,
            sample_mask=padded_mask,
            filter_type="butterworth",
            t_r=t_r,
            extrapolate=True
        )

    if padtype == "zero":
        filtered_data = butterworth(
            signals=padded_func_data,
            sampling_rate=1.0 / t_r,
            low_pass=low_pass,
            high_pass=high_pass,
        )[padlen:-padlen, :]  # remove 0-pad frames on both sides
        assert filtered_data.shape[0] == func_data.shape[0], "Filtered data must have the same number of timepoints as the original functional data"
    else:
        filtered_data = butterworth(
            signals=func_data,
            sampling_rate=1.0 / t_r,
            low_pass=low_pass,
            high_pass=high_pass,
            padtype=None if padtype == "none" else padtype,
            padlen=padlen
        )
    return filtered_data


class _FilterDataInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="Path to unfiltered timeseries (as a .nii, .nii.gz, or .dtseries.nii)."
    )
    in_mask = File(
        exists=True, mandatory=True,
        desc="Run mask (as a .txt)."
    )
    out_file = File(
        mandatory=True,
        desc="Path to filtered timeseries to create."
    )
    padtype = traits.Str(
        "zero",
        desc="Type of padding to use -- choices: 'odd', 'even', 'constant', 'zero', or 'none'"
    )
    padlen = traits.Int(
        50,
        desc="Length of pad."
    )


class _FilterDataOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(
        File(exists=True),
        desc="Filtered timeseries."
    )


class FilterData(SimpleInterface):
    """
    Generates a nuisance matrix for regression before final GLM.
    """

    input_spec = _FilterDataInputSpec
    output_spec = _FilterDataOutputSpec

    def _run_interface(self, runtime):
        img = nib.load(self.inputs.in_file)
        func_data = img.get_fdata()
        orig_shape = deepcopy(func_data.shape)
        if len(func_data.shape) == 2:  # cifti
            suffix = ".dtseries.nii"
        elif len(func_data.shape) == 4:  # nifti
            suffix = ".nii.gz"
            func_data = np.reshape(func_data, (-1, func_data.shape[-1])).T
        mask = np.loadtxt(self.inputs.in_mask)

        filtered_fdata = filter_data(
            func_data=func_data,
            mask=mask,
            t_r=self.inputs.t_r,
            low_pass=self.inputs.low_pass,
            high_pass=self.inputs.high_pass,
            padtype=self.inputs.padtype,
            padlen=self.inputs.padlen
        )
        if suffix == ".dtseries.nii":
            filtered_img = img.__class__(filtered_fdata,
                                         header=img.header,
                                         nifti_header=img.nifti_header)
            nib.save(filtered_img, self.inputs.out_file)
            self._results['out_file'] = self.inputs.out_file
        elif suffix == ".nii.gz":  # reshape back to 4d
            filtered_fdata = np.reshape(filtered_fdata.T, orig_shape)
            filtered_img = img.__class__(filtered_fdata, header=img.header)
            nib.save(filtered_img, self.inputs.out_file)
            self._results['out_file'] = self.inputs.out_file
        else:
            raise RuntimeError("Unexpected data type for functional data")
