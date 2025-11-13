from nipype import Node, Workflow, Function, MapNode
from nipype.pipeline.engine import Node as eNode
from nipype.interfaces.io import BIDSDataGrabber 
from nipype.interfaces.utility import IdentityInterface
from niworkflows.utils.bids import collect_participants
from niworkflows.interfaces.bids import DerivativesDataSink
from oceanproc.firstlevel.interfaces.nuisance import GenerateNuisanceMatrix
from oceanproc.firstlevel.interfaces.tmask import MakeTmask
from . import operations
from .interfaces import *
from nipype import config as ncfg
from pathlib import Path
# from .config import Options
# from bids.layout import BIDSFile
from bids.layout.utils import PaddedInt
from .config import all_opts
from . import utilities
from bids.utils import listify




def build_oceanfla_wf(task:str, subjects:str|list[str]|None, base_dir=Path|str):
    fla_wf = Workflow(name=f"task_{task}_wf", base_dir=base_dir)
    # fla_wf.base_dir = all_opts.work_dir

    subject_list = collect_participants(
        bids_dir=all_opts.preproc_layout,
        participant_label=listify(subjects)
    )
    
    start_node = Node(
        IdentityInterface(
            fields=["task"]
        ),
        name="task_start_node"
    )
    start_node.inputs.task = task

    for sub in subject_list:
        sessions = all_opts.preproc_layout.get_sessions(subject=sub)
        for ses in sessions:
            ses_wf = build_session_wf(subject=sub,
                                      session=ses,
                                      task=task)
            fla_wf.connect([
                (start_node, ses_wf, [
                    ("task", "inputnode.task")
                ])
            ])
    
    return fla_wf

    
    

def build_session_wf(subject, session, task):

    workflow = Workflow(name=f"sub_{subject}_ses_{session}_wf")

    input_node = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "task",
            ]
        ),
        name="inputnode"
    )
    input_node.inputs.subject = subject
    input_node.inputs.session = session

    space_run_info = utilities.parse_session_bold_files(layout=all_opts.preproc_layout,
                                                        subject=subject,
                                                        session=session,
                                                        task=task)
    space_dict = space_run_info[all_opts.func_space]
    func_space_wf = build_func_space_wf(func_space=all_opts.func_space,
                                        run_list=space_dict["runs"],
                                        file_extension=space_dict["extension"])
    
    workflow.connect([
        (input_node, func_space_wf, [
            ("subject", "inputnode.subject"),
            ("session", "inputnode.session"),
            ("task", "inputnode.task"),
        ])
    ])

    #### only doing one functional space at a time ###### 

    # for func_space, space_dict in space_run_info.items():
    #     func_space_wf = build_func_space_wf(func_space=func_space,
    #                                         run_list=space_dict["runs"],
    #                                         file_extension=space_dict["extension"])
        
    #     workflow.connect([
    #         (input_node, func_space_wf, [
    #             ("subject", "inputnode.subject"),
    #             ("session", "inputnode.session"),
    #             ("task", "inputnode.task"),
    #         ])
    #     ])

    ### DO Reporting Stuff for this subject ###

    return workflow




def build_func_space_wf(func_space:str, run_list:list[PaddedInt], file_extension:str):

    # Define the workflow and the input node for this functional space
    workflow = Workflow(name=f"space_{func_space}_wf")

    input_node = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "task",
            ]
        ),
        name="inputnode"
    )

    # Define the data grabber nodes to find the relevant files
    derivs_grabber = Node(
        BIDSDataGrabber(
            base_dir = all_opts.preproc_bids,
            datatype = 'func',
            
            output_query = {
                'bold': { 
                    'suffix': 'bold', 
                    'space': func_space,
                    'extension': file_extension,
                },
                'confounds': {
                    'suffix': 'timeseries',
                    'desc': 'confounds',
                    'extension': '.tsv',
                }
            },
            load_layout=all_opts.preproc_layout._root / ".bids_indexer"
        ), 
        name="derivs_bidssrc_node"
    )

    rawdata_grabber = Node(
        BIDSDataGrabber(
            base_dir = all_opts.raw_bids,
            datatype = 'func',
            output_query = {
                'events': {
                    'suffix': 'events',
                    'extension': '.tsv'
                },
            },
            load_layout=all_opts.raw_layout._root / ".bids_indexer"
        ),
        name="raw_bidssrc_node"
    )

    # Connect the inputs to the data-grabber nodes
    workflow.connect([
        (input_node, derivs_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ]),
        (input_node, rawdata_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ])
    ])

    # Create a run-level workflow for each run that has this functional space
    for run in sorted(run_list):
        run_level_wf = build_run_workflow(run=run)

        # Define a node to extract the run-specific files from the data-grabbers
        extract_run_group_node = Node(
            operations.ExtractRunGroup,
            name="extract_run_group_node"
        )
        # extract_run_group_node = Node( operations.ExtractRunGroup
        #     Function(
        #         function=operations.extract_run_group,
        #         input_names=["bold_list", "confounds_list", "events_list", "run_needed"],
        #         output_names=["bold_file",
        #                     "confounds_file",
        #                     "events_file"]
        #     ),
        #     name="extract_run_group_node"
        # )
        extract_run_group_node.inputs.run_needed = run
        
        # Connect the files to the run-level workflow
        workflow.connect([
            (derivs_grabber, extract_run_group_node, [
                ("bold", "bold_list"),
                ("confounds", "confounds_list")
            ]),
            (rawdata_grabber, extract_run_group_node, [
                ("events", "events_list"),
            ]),
            (extract_run_group_node, run_level_wf, [
                ("bold_file", "inputnode.bold_file"),
                ("confounds_file", "inputnode.confounds_file"),
                ("events_file", "inputnode.events_file"),
            ])
        ])
    

    ## DO STUFF AFTER THE RUN-LEVEL WORKFLOWS ###
    ##  * concat run-level info 
    ##  * run session-level glm

    return workflow



def build_run_workflow(run):

    # Define the workflow and the inputnode

    workflow = Workflow(name=f"run_{run}_wf")

    inputnode = Node(
        IdentityInterface(
            fields=[
                "bold_file",
                "confounds_file",
                "events_file",
            ]
        ),
        name = "inputnode"
    )

    outputnode = Node(
        IdentityInterface(
            fields=[
                "cleaned_bold",
                "design_matrix",
                "nuisance_matrix",
                "tmask",
            ]
        ),
        name = "outputnode"
    )


    tr_extract_func = lambda bids_file: bids_file.entities["RepetitionTime"]
    extract_tr_node = Node(
        Function(
            input_names=["bids_file"],
            output_names=["tr"],
            function=tr_extract_func
        ),
        name = "tr_extract_node"
    )

    get_volumes_node = Node(
        operations.GetVolumeCount,
        name="get_run_volumes_node"
    )
    get_volumes_node.inputs.brain_mask = all_opts.brain_mask

    events_matrix_node = Node(
        operations.EventsMatrix(
            fir = all_opts.fir,
            hrf = all_opts.hrf,
            fir_vars = all_opts.fir_vars,
            hrf_vars = all_opts.hrf_vars,
            unmodeled = all_opts.unmodeled,
        ),
        name="events_matix_node"
    )

    nuisance_mat_node = Node(
        GenerateNuisanceMatrix(
            confounds_columns = all_opts.confounds,
            demean = True, # not all_opts.exclude_run_mean
            linear_trend = True, # not all_opts.exclude_run_trend
            spike_threshold = all_opts.fd_threshold if all_opts.spike_regression else None,
            volterra_lag = all_opts.volterra_lag,
            volterra_columns = all_opts.volterra_columns,
        )
    )

    tmask_node = Node(
        MakeTmask(
            fd_threshold = all_opts.fd_threshold,
            minimum_unmasked_neighbors = all_opts.minimum_unmasked_neighbors,
            start_censoring = all_opts.start_censoring
        ),
        name="make_tmask_node"
    )

    workflow.connect([
        (inputnode, extract_tr_node, [
            ("bold_file", "bids_file")
        ]),
        (inputnode, get_volumes_node, [
            ("bold_file", "bold_in")
        ]), 
        (extract_tr_node, events_matrix_node, [
            ("tr", "tr")
        ]),
        (get_volumes_node, events_matrix_node, [
            ("volumes", "volumes")
        ]),
        (inputnode, events_matrix_node, [
            ("events_file", "event_file")
        ]),
        (inputnode, nuisance_mat_node, [
            ("confounds_file", "confounds_file")
        ]),
        (inputnode, tmask_node, [
            ("confounds_file", "confounds_file")
        ])
    ])


    if all_opts.highpass or all_opts.lowpass:
        if "mean" not in all_opts.nuisance_regression:
            # logger.warning("High-, low-, or band-pass specified, but mean not specified as nuisance regressor -- adding this in automatically")
            all_opts.nuisance_regression.append("mean")
        if "trend" not in all_opts.nuisance_regression:
            # logger.warning("High-, low-, or band-pass specified, but trend not specified as nuisance regressor -- adding this in automatically")
            all_opts.nuisance_regression.append("trend")
    
    if all_opts.nuisance_regression:
        pass



    return workflow



