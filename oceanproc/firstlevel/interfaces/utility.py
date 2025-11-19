from nipype.interfaces.utility.base import MergeInputSpec, _ravel
from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    DynamicTraitedSpec,
    isdefined,
    traits,
)
from bids.utils import listify

class MergeUniqueInputSpec(MergeInputSpec):
    


class MergeUniqueOutputSpec(DynamicTraitedSpec):
    pass

class MergeUnique(SimpleInterface):
    input_spec = MergeInputSpec
    output_spec = DynamicTraitedSpec
    _sep = "_x"

    def _run_interface(self, runtime):
        input_keys = [k.split(self._sep) for k in self.inputs.get().keys()]
        input_key_name_set = set([t[0] for t in input_keys if len(t)==2 and t[1].isnumeric()])
        if len(input_key_name_set) < 1:
            return runtime
        max_index = max([int(t[1]) for t in input_keys if t[0] in input_key_name_set])
        print(input_key_name_set)
        print(max_index)
        for input_key in input_key_name_set:
            out = []
            values = [getattr(self.inputs, f"{input_key}{self._sep}{idx}") 
                      for idx in range(1, max_index + 1) 
                      if hasattr(self.inputs,  f"{input_key}{self._sep}{idx}")]
            if self.inputs.axis == "vstack":
                for value in values:
                    if isinstance(value, list) and not self.inputs.no_flatten:
                        out.extend(_ravel(value) if self.inputs.ravel_inputs else value)
                    else:
                        out.append(value)
            else:
                lists = [listify(val) if val is not None else [None] for val in values]
                out = [[val[i] for val in lists] for i in range(len(lists[0]))]
            if all([o is None for o in out]):
                out = None
            print(out)
            self._results[input_key] = out
        return runtime
