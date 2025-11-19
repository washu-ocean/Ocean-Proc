from nipype.interfaces.utility.base import MergeInputSpec, _ravel
from nipype.interfaces.io import add_traits, IOBase
from nipype.interfaces.io import BIDSDataGrabber, BIDSDataGrabberInputSpec
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
    Undefined
)
from bids.utils import listify
from sqlalchemy import desc


class MergeUniqueOutputSpec(DynamicTraitedSpec):
    pass

class MergeUnique(IOBase):
    input_spec = MergeInputSpec
    output_spec = DynamicTraitedSpec
    _sep = "_x"

    # def _run_interface(self, runtime):
    def _list_outputs(self):
        outputs = self._outputs().get()
        # return super()._list_outputs()
        input_keys = [k.split(self._sep) for k in self.inputs.get().keys()]
        input_key_name_set = set([t[0] for t in input_keys if len(t)==2 and t[1].isnumeric()])
        if len(input_key_name_set) < 1:
            return outputs
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
            outputs[input_key] = out
        return self._results




class BidsDataGrabberExtInputSpec(BIDSDataGrabberInputSpec):
    validate = traits.Bool(
        default_value=False,
        desc="If the bids directory should be validated"
    )

    allow_invalid_filters = traits.Bool(
        default_value=True,
        desc="If invalid bids filters are allowed"
    )

    is_derivative = traits.Bool(
        default_value=False,
        desc=""
    )

class BidsDataGrabberExt(BIDSDataGrabber):
    input_spec = BidsDataGrabberExtInputSpec

    def _list_outputs(self):
        from bids import BIDSLayout

        # if load_layout is given load layout which is on some datasets much faster
        if isdefined(self.inputs.load_layout):
            layout = BIDSLayout(database_path=self.inputs.load_layout, validate=self.inputs.validate, is_derivative=self.inputs.is_derivative)
        else:
            layout = BIDSLayout(
                self.inputs.base_dir, derivatives=self.inputs.index_derivatives, validate=self.inputs.validate, is_derivative=self.inputs.is_derivative
            )

        if isdefined(self.inputs.extra_derivatives):
            layout.add_derivatives(self.inputs.extra_derivatives)

        # If infield is not given nm input value, silently ignore
        filters = {}
        for key in self._infields:
            value = getattr(self.inputs, key)
            if isdefined(value):
                filters[key] = value

        outputs = {}
        for key, query in self.inputs.output_query.items():
            args = query.copy()
            args.update(filters)
            filelist = layout.get(return_type="file", **args)
            if len(filelist) == 0:
                msg = "Output key: %s returned no files" % key
                if self.inputs.raise_on_empty:
                    raise OSError(msg)
                else:
                    # iflogger.warning(msg)
                    filelist = Undefined

            outputs[key] = filelist
        return outputs