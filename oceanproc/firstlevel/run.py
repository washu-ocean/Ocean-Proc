# from nipype import Node, Workflow, Function, MapNode
# from nipype.pipeline.engine import Node as eNode
# from nipype.interfaces.io import BIDSDataGrabber 
# from . import operations
from nipype import config as ncfg
# from pathlib import Path
from .parser import parse_args
from bids.utils import listify
# import bids
# from bids.layout.utils import PaddedInt





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


# def parse_session_bold_files(layout:bids.BIDSLayout, subject:str, session:str, task:str):
#     files = layout.get(subject=subject, session=session, task=task, suffix="bold", datatype="func", extensions=[".nii",".nii.gz",".dtseries.nii"])
#     space_run_dict = dict()
#     for f in files:
#         run = f.entities["run"] if "run" in f.entities else PaddedInt('01')
#         space = f.entities["space"] if "space" in f.entities else "func"
#         if space in space_run_dict:
#             space_run_dict[space]["runs"].append(run)
#         else:
#             space_run_dict[space] = {}
#             space_run_dict[space]["extension"] = f.entities["extension"]
#             space_run_dict[space]["runs"] = [run]
#     return space_run_dict


def main():

    work_dir="/data/sylvester/data1/users/agardr"
    parse_args()
    func_space = ""
    extension = ""
    from .config import all_opts
    from .workflows import build_oceanfla_wf
    
    oceanfla_wf = build_oceanfla_wf(
        task=listify(all_opts.task),
        subjects=all_opts.subject,
        base_dir=work_dir
    )
    
    oceanfla_wf.run()
    # space_run_info = parse_session_bold_files(
    #     layout=all_opts.preproc_layout,
    #     subject=all_opts.subject,
    #     session=all_opts.session, 
    #     task=all_opts.task
    # )

    # print_node = Node(
    #     Function(
    #         function=print_me,
    #         input_names=["x"],
    #         output_names=[]
    #     ),
    #     name="print"
    # )
    # in_file = "/data/sylvester/data1/users/agardr/sub-4000_ses-030321_task-oddball_run-01_desc-b07_events.tsv"
    # rawdata_dir = Path("/data/sylvester/data1/datasets/npad_renewal_pilot/rawdata")
    # deriv_dir = Path("/data/sylvester/data1/datasets/npad_renewal_pilot/derivatives/fmriprep")
    # derivs_grabber = Node(
    #     BIDSDataGrabber(
    #         base_dir = all_opts.preproc_bids,
    #         subject = all_opts.subject,
    #         session = all_opts.session,
    #         task = all_opts.task,
    #         datatype = 'func',
    #         output_query = {
    #             'bold': { 
    #                 'suffix': 'bold', 
    #                 'space': func_space,
    #                 'extension': extension,
    #             },
    #             'confounds': {
    #                 'suffix': 'timeseries',
    #                 'desc': 'confounds',
    #                 'extension': '.tsv',
    #             }
    #         },
    #         load_layout=all_opts.preproc_layout._root / ".bids_indexer"
    #     ), 
    #     name="deriv_grabber"
    # )

    # rawdata_grabber = Node(
    #     BIDSDataGrabber(
    #         base_dir = all_opts.raw_bids,
    #         subject = all_opts.subject,
    #         session = all_opts.session,
    #         task = all_opts.task,
    #         datatype = 'func',
    #         output_query = {
    #             'events': {
    #                 'suffix': 'events',
    #                 'extension': '.tsv'
    #             },
    #         },
    #         load_layout=all_opts.raw_layout._root / ".bids_indexer"
    #     ),
    #     name="raw_grabber"
    # )

    # group_files_node = Node(
    #     Function(
    #         function=operations.group_runs,
    #         input_names=["bolds", "confounds", "events"],
    #         output_names=["run_list"]
    #     ),
    #     name="group_runs_node"
    # )

    # extract_run_files_node = Node(
    #     Function(
    #         function=operations.extract_run_files,
    #         input_names=["run_dict"],
    #         output_names=["bold",
    #                       "confounds",
    #                       "events"]
    #     ),
    #     name="extract_run_files"
    # )
    # extract_run_files_node.iterables = [('run_dict', space_run_info[func_space]["runs"])]

    # run_sub_wf = build_run_workflow()

    # make_design = Node(
    #     operations.DesignMat(
    #         hrf = all_opts.hrf,
    #         fir = all_opts.fir,
    #         hrf_vars = all_opts.hrf_vars,
    #         fir_vars = all_opts.fir_vars,
    #         unmodeled = all_opts.unmodeled

    #     ), 
    #     name="make_design", 
    #     iterables=["in_file"]
    # )
    

    # identity_func = lambda in_file: str(in_file)
    # info_node = eNode(name="get_file",
    #     interface=Function(input_names=["in_file"],
    #             output_names=["out_file"],
    #             function=identity_func))
    # make_design = Node(operations.DesignMat(in_file=in_file,
    #                                         hrf=[5,12],
    #                                         volumes=335,
    #                                         tr=1.2),
    #                     name="make_design")
    
    # test_wf = Workflow(name="tester", base_dir=work_dir)
    # test_wf.connect(derivs_grabber, "bold", group_files_node, "bolds")
    # test_wf.connect(derivs_grabber, "confounds", group_files_node, "confounds")
    # test_wf.connect(rawdata_grabber, "events", group_files_node, "events")
    # test_wf.connect(group_files_node, "run_list", extract_run_files_node, "run_dict")
    # test_wf.connect(extract_run_files_node, "events", print_node, "x")
    # test_wf.run()
