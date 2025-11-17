# from nipype import logging
import sys
import datetime
from pathlib import Path
import bids

all_opts = None

class Options():
    _instance = None
    _initialized = False
    layouts = []
    generic_nuisance_columns = ["mean", "trend", "spike"]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, opts=None):
        if not self._initialized and opts:
            for k,v in opts.items():
                if isinstance(v, bids.BIDSLayout):
                    self.layouts.append(v)
                setattr(self, k, v)
            global all_opts 
            all_opts = self
            
# class loggers():

#     _default_log_format = "%(levelname)s:%(asctime)s:%(module)s: %(message)s"
#     _log_level = logging.INFO
#     _stream_handler = logging.StreamHandler(stream=sys.stdout)

#     root = None
#     operations = None
#     utils = None

#     @classmethod
#     def initialize(cls):
#         opts = Options()
#         if opts.debug:
#             cls._log_level = logging.DEBUG

#         log_dir = opts.output_dir.parent / "logs"
#         log_dir.mkdir(parents=True, exist_ok=True)

#         log_path = log_dir / f"{opts.file_name_base}_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}{opts.custom_desc}.log"
#         cls._file_handler = logging.FileHandler(log_path)

#         logging.basicConfig(level=cls._log_level,
#                     handlers=[
#                         cls._stream_handler,
#                         cls._file_handler
#                     ],
#                     format=cls._default_log_format)
        
#         cls.root = logging.getLogger()
#         cls.operations = logging.getLogger("first_level.operations")
#         cls.utils = logging.getLogger("first_level.utils")
        


        
def set_configs(args):
    Options(args) 
    # loggers.initialize()



def get_layout_for_file(file) -> bids.BIDSLayout:
    if isinstance(file, Path):
        file = str(file.resolve())
    if isinstance(file, str):
        file = str(Path(file).resolve())
    else:
        raise ValueError(f"argument must be of type Path or str, not {type(file)}")

    for lay in all_opts.layouts:
        if file.startswith(str(lay._root.resolve())):
            return lay
    raise RuntimeError(f"No layout correspond to the input file {file}")


def get_bids_file(file):
    file_layout = get_layout_for_file(file)
    return file_layout.get_file(file)