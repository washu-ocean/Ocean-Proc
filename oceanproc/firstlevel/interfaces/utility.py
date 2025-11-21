from nipype.interfaces.utility.base import MergeInputSpec, _ravel
from nipype.interfaces.io import IOBase
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    DynamicTraitedSpec,
    traits,
)
from bids.utils import listify



class MergeUnique(IOBase):
    input_spec = MergeInputSpec
    output_spec = DynamicTraitedSpec
    _sep = "_x"

    # def _run_interface(self, runtime):
    def _list_outputs(self):
        outputs = self._outputs().get()
        # return super()._list_outputs()
        input_keys = [k.split(self._sep) for k in self.inputs.get().keys()]
        input_key_name_set = set(
            [t[0] for t in input_keys if len(t) == 2 and t[1].isnumeric()])
        if len(input_key_name_set) < 1:
            return outputs
        max_index = max([int(t[1])
                        for t in input_keys if t[0] in input_key_name_set])
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
                        out.extend(
                            _ravel(value) if self.inputs.ravel_inputs else value)
                    else:
                        out.append(value)
            else:
                lists = [listify(val) if val is not None else [None]
                         for val in values]
                out = [[val[i] for val in lists] for i in range(len(lists[0]))]
            if all([o is None for o in out]):
                out = None
            print(out)
            outputs[input_key] = out
        return outputs


class ExtractDataGroupInputSpec(BaseInterfaceInputSpec):
    bold_list = traits.List(
        trait=traits.File(exists=True),
        desc="list of bold files"
    )

    confounds_list = traits.List(
        trait=traits.File(exists=True),
        desc="list of confounds files"
    )

    events_list = traits.List(
        trait=traits.File(exists=True),
        desc="list of event files"
    )

    task = traits.Str(desc="The task name of the data")

    run = traits.Int(desc="The run number of the data")


class ExtractDataGroupOutputSpec(TraitedSpec):
    bold_file = traits.File(
        exists=True,
        desc="The described bold file"
    )

    confounds_file = traits.File(
        exists=True,
        desc="The described confounds file"
    )

    events_file = traits.File(
        exists=True,
        desc="The described events file"
    )


class ExtractDataGroup(SimpleInterface):
    input_spec = ExtractDataGroupInputSpec
    output_spec = ExtractDataGroupOutputSpec

    def _run_interface(self, runtime):

        bold_file, confounds_file, events_file = extract_task_run_group(
            bold_list=self.inputs.bold_list,
            confounds_list=self.inputs.confounds_list,
            events_list=self.inputs.events_list,
            task_needed=self.inputs.task,
            run_needed=self.inputs.run,

        )

        self._results["bold_file"] = bold_file
        self._results["confounds_file"] = confounds_file
        self._results["events_file"] = events_file

        return runtime


def extract_task_run_group(bold_list: list,
                           confounds_list: list,
                           events_list: list,
                           task_needed: str,
                           run_needed: int):
    from oceanproc.firstlevel.config import get_bids_file
    run_dict = {
        "bold": None,
        "confounds": None,
        "events": None
    }
    for ftype, file_list in {"bold": bold_list, "confounds": confounds_list, "events": events_list}.items():
        for file in file_list:
            bids_file = get_bids_file(file)
            run = int(bids_file.entities.get("run", 1))
            if run == int(run_needed) and bids_file.entities["task"] == task_needed:
                run_dict[ftype] = bids_file
                break

    if not all(list(run_dict.values())):
        raise RuntimeError(
            f"Could not find all the needed files for run-{run_needed}")

    return run_dict["bold"], run_dict["confounds"], run_dict["events"]
