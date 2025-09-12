from nipype import Node, Workflow, Function
from nipype.pipeline.engine import Node as eNode
from . import operations
from nipype import config as ncfg



ncfg.update_config(
    {
        'execution': {
            'crashfile_format': "txt",
            'stop_on_first_crash': True,
        }
    }
)

def main():

    in_file = "/data/sylvester/data1/users/agardr/sub-4000_ses-030321_task-oddball_run-01_desc-b07_events.tsv"

    identity_func = lambda in_file: str(in_file)
    info_node = eNode(name="get_file",
        interface=Function(input_names=["in_file"],
                output_names=["out_file"],
                function=identity_func))
    make_design = Node(operations.DesignMat(in_file=in_file,
                                            hrf=[5,12],
                                            volumes=335,
                                            tr=1.2),
                        name="make_design")
    
    test_wf = Workflow(name="tester", base_dir="/data/sylvester/data1/users/agardr")

    test_wf.connect(make_design, "out_file", info_node, "in_file")
    test_wf.run()
