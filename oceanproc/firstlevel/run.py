from nipype import Node, Workflow, Function, MapNode
from nipype.pipeline.engine import Node as eNode
from nipype.interfaces.io import BIDSDataGrabber 
from . import operations
from nipype import config as ncfg
from pathlib import Path
from parser import parse_args
from config import all_opts



ncfg.update_config(
    {
        'execution': {
            'crashfile_format': "txt",
            'stop_on_first_crash': True,
        }
    }
)


def print_me(x):
    print(f"Printing stuff:\n{x}\n")


def main():

    work_dir="/data/sylvester/data1/users/agardr"
    parse_args()
    opts = all_opts


    print_node = Node(
        Function(
            function=print_me,
            input_names=["x"],
            output_names=[]
        ),
        name="print"
    )
    # in_file = "/data/sylvester/data1/users/agardr/sub-4000_ses-030321_task-oddball_run-01_desc-b07_events.tsv"
    # rawdata_dir = Path("/data/sylvester/data1/datasets/npad_renewal_pilot/rawdata")
    # deriv_dir = Path("/data/sylvester/data1/datasets/npad_renewal_pilot/derivatives/fmriprep")
    derivs_grabber = Node(
        BIDSDataGrabber(
            base_dir = opts.preproc_bids,
            subject = opts.subject,
            session = opts.session,
            task = opts.task,
            datatype = 'func',
            output_query = {
                'bold': { 
                    'suffix': 'bold', 
                    'extension': opts.bold_file_type,
                },
                'confounds': {
                    'suffix': 'timeseries',
                    'desc': 'confounds',
                    'extension': '.tsv',
                }
            },
            load_layout=opts.preproc_layout._root / ".bids_indexer"
        ), 
        name="deriv_grabber"
    )

    rawdata_grabber = Node(
        BIDSDataGrabber(
            base_dir = opts.raw_bids,
            subject = opts.subject,
            session = opts.session,
            task = opts.task,
            datatype = 'func',
            output_query = {
                'events': {
                    'suffix': 'events',
                    'extension': '.tsv'
                },
            },
            load_layout=opts.raw_layout._root / ".bids_indexer"
        ),
        name="raw_grabber"
    )

    group_files_node = Node(
        Function(
            function=operations.group_runs,
            input_names=["bolds", "confounds", "events"],
            output_names=["run_d"]
        ),
        name="group_runs"
    )

    make_design = Node(
        operations.DesignMat(
            hrf = opts.hrf,
            fir = opts.fir,
            hrf_vars = opts.hrf_vars,
            fir_vars = opts.fir_vars,
            unmodeled = opts.unmodeled

        ), 
        name="make_design", 
        iterables=["in_file"]
    )
    

    identity_func = lambda in_file: str(in_file)
    info_node = eNode(name="get_file",
        interface=Function(input_names=["in_file"],
                output_names=["out_file"],
                function=identity_func))
    # make_design = Node(operations.DesignMat(in_file=in_file,
    #                                         hrf=[5,12],
    #                                         volumes=335,
    #                                         tr=1.2),
    #                     name="make_design")
    
    test_wf = Workflow(name="tester", base_dir=work_dir)
    test_wf.connect(derivs_grabber, "bold", group_files_node, "bolds")
    test_wf.connect(derivs_grabber, "confounds", group_files_node, "confounds")
    test_wf.connect(rawdata_grabber, "events", group_files_node, "events")
    test_wf.connect(group_files_node, "run_dict", print_node, "x")
    test_wf.run()
