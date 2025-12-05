from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype import Function


class CheckRunRetentionInputSpec(BaseInterfaceInputSpec):
    tmask_file = File(
        exists=True,
        mandatory=True,
        desc="Path to the tmask file"
    )

    retention_threshold = traits.Float(
        default_value=0.0,
        desc="The percentage of frames that must be unmasked, excluding the frames removed by start censoring"
    )

    start_censoring = traits.Int(
        0,
        desc="Number of frames to censor out automatically at the beginning of each run."
    )


class CheckRunRetentionOutputSpec(TraitedSpec):
    valid = traits.Bool(
        default_value=False,
        desc="If this run passed the frame retention check"
    )


class CheckRunRetention(SimpleInterface):
    input_spec = CheckRunRetentionInputSpec
    output_spec = CheckRunRetentionOutputSpec

    def _run_interface(self, runtime):

        self._results["valid"] = check_run_retention(
            tmask_file=self.inputs.tmask_file,
            retention_threshold=self.inputs.retention_threshold,
            start_censoring=self.inputs.start_censoring
        )

        return runtime


class CheckRuntSNRInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc="Path to the bold file"
    )

    tsnr_threshold = traits.Float(
        default_value=0.0,
        desc="The lowest, whole brain tSNR value allowed"
    )

    tmask_file = traits.Union(
        File(exists=True),
        None,
        default_value=None,
        desc="Path to the tmask file"
    )

    brain_mask = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )


class CheckRuntSNROutputSpec(TraitedSpec):
    valid = traits.Bool(
        default_value=False,
        desc="If this run passed the tSNR check"
    )


class CheckRuntSNR(SimpleInterface):
    input_spec = CheckRuntSNRInputSpec
    output_spec = CheckRuntSNROutputSpec

    def _run_interface(self, runtime):

        self._results["valid"] = check_run_tsnr(
            bold_file=self.inputs.bold_file,
            tsnr_threshold=self.inputs.tsnr_threshold,
            tmask_file=self.inputs.tmask_file,
            brain_mask=self.inputs.brain_mask
        )
        return runtime


CheckValidations = Function(
    function=all,
    input_names=["validation_list"],
    output_names="include"
)


def check_run_retention(tmask_file: Path | str,
                        retention_threshold: float,
                        start_censoring: int):
    import numpy as np

    tmask_data = np.loadtxt(tmask_file)[start_censoring:]
    total_frames = tmask_data.shape[0]
    retained_frames = np.sum(tmask_data)
    perc_retained = (retained_frames / total_frames) * 100

    return perc_retained >= retention_threshold


def check_run_tsnr(bold_file: str | Path,
                   tsnr_threshold: float,
                   tmask_file: str | Path = None,
                   brain_mask: str = None):
    from ..utilities import load_data
    import numpy as np

    bold_data = load_data(bold_file, brain_mask)
    if tmask_file:
        tmask_data = np.loadtxt(tmask_file).astype(bool)
        bold_data = bold_data[tmask_data, :]

    tsnr_map = np.nanmean(bold_data, axis=0) / np.nanstd(bold_data, axis=0)
    whole_brain_avg_tsnr = np.nanmean(tsnr_map)
    return whole_brain_avg_tsnr >= tsnr_threshold
