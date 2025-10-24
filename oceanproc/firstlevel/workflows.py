from nipype import Node, Workflow, Function, MapNode
from nipype.pipeline.engine import Node as eNode
from nipype.interfaces.io import BIDSDataGrabber 
from . import operations
from nipype import config as ncfg
from pathlib import Path
from .config import Options


# def build_run_workflow():