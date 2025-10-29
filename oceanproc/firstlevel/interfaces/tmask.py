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

class _MakeTmaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Path to nuisance matrix (as a .csv or .tsv)"
    )
    out_file = File(
        mandatory=True,
        desc="Path to tmask (a .txt file)"
    )
    minimum_unmasked_neighbors = traits.Int(
        0,
        desc="""\
Number of frames to mask out on either side of each frame masked
due to motion.
"""
    )


class _MakeTmaskOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Path to tmask (a .txt file)"
    )


class MakeTmask(SimpleInterface):
    input_spec = _MakeTmaskInputSpec
    output_spec = _MakeTmaskOutputSpec


    def _run_interface(self, runtime):
        pass    


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = getattr(self, '_result')
        return outputs


