from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


def make_tmask(confounds_file: Path | str,
               fd_threshold: int,
               minimum_unmasked_neighbors: int,
               start_censoring: int):
    from ..utilities import replace_entities
    import pandas as pd
    import numpy as np

    if start_censoring < 0:
        raise ValueError("The 'start_censoring' argument of make_tmask() must be 0 or positive.")
    if minimum_unmasked_neighbors < 0:
        raise ValueError("The 'minimum_unmasked_neighbors' argument of make_tmask() must be 0 or positive.")
    if fd_threshold < 0:
        raise ValueError("The 'fd_threshold' argument of make_tmask() must be 0 or positive.")

    df = pd.read_csv(confounds_file, sep="\t")

    fd_arr = df.loc[:, "framewise_displacement"].to_numpy()
    if minimum_unmasked_neighbors > 0:
        fd_arr_padded = np.pad(fd_arr, pad_width := minimum_unmasked_neighbors)
        fd_mask = np.full(len(fd_arr_padded), False)
        for i in range(pad_width, len(fd_arr_padded) - pad_width):
            if all(fd_arr_padded[range(i - pad_width, i + pad_width + 1)] < fd_threshold):
                fd_mask[i] = True
            elif i - pad_width < pad_width and all(fd_arr_padded[range(pad_width, i + pad_width + 1)] < fd_threshold):
                fd_mask[i] = True
            elif i + pad_width + 1 > len(fd_arr_padded) - pad_width and all(fd_arr_padded[range(i - pad_width, len(fd_arr_padded) - pad_width)] < fd_threshold):
                fd_mask[i] = True
            else:
                fd_mask[i] = False
        fd_mask = fd_mask[pad_width:-pad_width]
    else:
        fd_mask = fd_arr < fd_threshold
    fd_mask[:start_censoring] = False

    out_file = replace_entities(
            file=confounds_file, 
            entities={
                "suffix": f"{str(fd_threshold).replace('.', 'p')}mm-tmask", 
                "ext": ".txt",
                "path": None
        })
    
    np.savetxt(out_file, fd_mask)
    return out_file


class _MakeTmaskInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="Path to nuisance matrix (as a .csv or .tsv)"
    )
    # out_file = File(
    #     mandatory=True,
    #     desc="Path to tmask (a .txt file)"
    # )
    fd_threshold = traits.Float(
        mandatory=True,
        desc="FD threshold for masking frames."
    )
    minimum_unmasked_neighbors = traits.Int(
        0,
        desc="""\
Number of frames to mask out on either side of each frame masked
due to motion.
"""
    )
    start_censoring = traits.Int(
        0,
        desc="Number of frames to censor out automatically at the beginning of each run."
    )


class _MakeTmaskOutputSpec(TraitedSpec):
    tmask_file = File(
        exists=True,
        desc="Path to tmask (a .txt file)"
    )


class MakeTmask(SimpleInterface):
    input_spec = _MakeTmaskInputSpec
    output_spec = _MakeTmaskOutputSpec

    def _run_interface(self, runtime):
        
        self._results["tmask_file"] = make_tmask(
            confounds_file=self.inputs.confounds_file,
            fd_threshold=self.inputs.fd_threshold,
            minimum_unmasked_neighbors=self.inputs.minimum_unmasked_neighbors,
            start_censoring=self.inputs.start_censoring,
        )

        return runtime

    # def _list_outputs(self):
    #     outputs = self._outputs().get()
    #     outputs['out_file'] = getattr(self, '_results')
    #     return outputs
