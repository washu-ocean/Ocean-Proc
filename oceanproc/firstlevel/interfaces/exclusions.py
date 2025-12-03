from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


def check_run_retention(tmask_file: Path|str,
                        retention_threshold: float,
                        start_censoring: int):
    import numpy as np
    
    tmask_data = np.loadtxt(tmask_file)[start_censoring:]
    total_frames = tmask_data.shape[0]
    retained_frames = np.sum(tmask_data)
    perc_retained = (retained_frames / total_frames) * 100

    return perc_retained < retention_threshold

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
    exclude = traits.Bool(
        default_value=False,
        desc="If this run needs to be excluded from the glm or not"
    )

class CheckRunRetention(SimpleInterface):
    input_spec = CheckRunRetentionInputSpec
    output_spec = CheckRunRetentionOutputSpec

    def _run_interface(self, runtime):

        self._results["exclude"] = check_run_retention(
            tmask_file=self.inputs.tmask_file,
            retention_threshold=self.inputs.retention_threshold,
            start_censoring=self.inputs.start_censoring
        )

        return runtime
    


# class CheckRuntSNRInputSpec(BaseInterfaceInputSpec):
#     tmask_file = File(
#         exists=True,
#         mandatory=True,
#         desc="Path to the tmask file"
#     )
#     retention_threshold = traits.Float(
#         default_value=0.0,
#         desc="The percentage of frames that must be unmasked, excluding the frames removed by start censoring"
#     )
#     start_censoring = traits.Int(
#         0,
#         desc="Number of frames to censor out automatically at the beginning of each run."
#     )

# class CheckRuntSNROutputSpec(TraitedSpec):
#     exclude = traits.Bool(
#         default_value=False,
#         desc="If this run needs to be excluded from the glm or not"
#     )

# class CheckRuntSNR(SimpleInterface):
#     input_spec = CheckRuntSNRInputSpec
#     output_spec = CheckRuntSNROutputSpec

#     def _run_interface(self, runtime):

#         self._results["exclude"] = check_run_retention(
#             tmask_file=self.inputs.tmask_file,
#             retention_threshold=self.inputs.retention_threshold,
#             start_censoring=self.inputs.start_censoring
#         )

#         return runtime

