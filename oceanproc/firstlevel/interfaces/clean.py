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
from nilearn.signal import (
    butterworth,
    _handle_scrubbed_volumes
)
from ..utilities import load_data, create_image_like
from nipype.utils.filemanip import split_filename, fname_presuffix


class _FilterDataInputSpec(BaseInterfaceInputSpec):
    func_file = File(
        exists=True, mandatory=True,
        desc="Path to unfiltered timeseries (as a .nii, .nii.gz, or .dtseries.nii)."
    )
    in_mask = File(
        exists=True, mandatory=True,
        desc="Run mask (as a .txt)."
    )
    high_pass = traits.Float(
        default_value=0.008,
        desc="The lowest frequency allowed (Hz)"
    )
    low_pass = traits.Float(
        default_value=0.1,
        desc="The highest frequency allowed (Hz)"
    )
    tr = traits.Float(
        desc="The Repetition Time for this BOLD run"
    )
    padtype = traits.Str(
        "zero",
        desc="Type of padding to use -- choices: 'odd', 'even', 'constant', 'zero', or 'none'"
    )
    padlen = traits.Int(
        50,
        desc="Length of pad."
    )
    brain_mask = traits.Union( 
        traits.File(exists=True),
        None, 
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
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
        mask = np.loadtxt(self.inputs.in_mask).astype(bool)
        filtered_fdata = filter_data(
            func_data= load_data(self.inputs.func_file, self.inputs.brain_mask),
            mask=mask,
            tr=self.inputs.tr,
            low_pass=self.inputs.low_pass,
            high_pass=self.inputs.high_pass,
            padtype=self.inputs.padtype,
            padlen=self.inputs.padlen
        )
        _, fname, ext = split_filename(self.inputs.func_file)
        out_path = f"{runtime.cwd}/{fname}_filtered{ext}"
        create_image_like(data=filtered_fdata, 
                          source_header=self.inputs.func_file, 
                          out_file=out_path,
                          brain_mask=self.inputs.brain_mask)
        self._results["out_file"] = out_path
        return runtime

    # def _list_outputs(self):
    #     return {'out_file': self.inputs.out_file}



class PercentChangeInputSpec(BaseInterfaceInputSpec):
    func_file = File(exists=True, mandatory=True, 
                        desc="A BIDS style bold file")
    
    in_mask = File(
        exists=True, mandatory=True,
        desc="Run mask (as a .txt)."
    )
    
    brain_mask = traits.Union( 
        traits.File(exists=True),
        None, 
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )
    

class PercentChangeOutputSpec(TraitedSpec):
    out_file = File(exists=True, 
                    desc="The functional data after a percent signal change transformation")
    

class PercentChange(SimpleInterface):
    input_spec = PercentChangeInputSpec
    output_spec = PercentChangeOutputSpec

    def _run_interface(self, runtime):
        from ..utilities import replace_entities
        mask = np.loadtxt(self.inputs.in_mask).astype(bool)
        psc_data = percent_signal_change(
            data = load_data(self.inputs.func_file, self.inputs.brain_mask),
            mask = mask
        )
        
        out_path = replace_entities(
            file=self.inputs.func_file,
            entities={"desc":"percent-change", "path":None}
        )
        create_image_like(data=psc_data, 
                          source_header=self.inputs.func_file, 
                          out_file=out_path,
                          brain_mask=self.inputs.brain_mask)
        self._results["out_file"] = out_path
        return runtime




def filter_data(func_data: npt.ArrayLike,
                mask: npt.ArrayLike,
                tr: float,
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
            t_r=tr,
            extrapolate=True
        )
    
    filtered_data = butterworth(
        signals=padded_func_data,
        sampling_rate=1.0 / tr,
        low_pass=low_pass,
        high_pass=high_pass,
        padtype=None if padtype == "none" else padtype,
    )[padlen:-padlen, :]  # remove 0-pad frames on both sides

    assert filtered_data.shape[0] == func_data.shape[0], "Filtered data must have the same number of timepoints as the original functional data"
    return filtered_data



def percent_signal_change(data: npt.ArrayLike, mask: npt.ArrayLike):
    masked_data = data[mask, :]
    mean = np.nanmean(masked_data, axis=0)
    mean = np.repeat(mean[np.newaxis,:], data.shape[0], axis=0)
    psc_data = ((data - mean) / np.abs(mean)) * 100
    non_valid_indices = np.where(~np.isfinite(psc_data))
    if len(non_valid_indices[0]) > 0:
        # logger.warning("Found vertices with zero signal, setting these to zero")
        psc_data[np.where(~np.isfinite(psc_data))] = 0
    return psc_data