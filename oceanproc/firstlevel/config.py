import logging
import sys
import datetime

class Options():
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, opts=None):
        if not self._initialized and opts:
            for k,v in opts.items():
                setattr(self, k, v)
            
class loggers():

    _default_log_format = "%(levelname)s:%(asctime)s:%(module)s: %(message)s"
    _log_level = logging.INFO
    _stream_handler = logging.StreamHandler(stream=sys.stdout)

    root = None
    operations = None
    utils = None

    @classmethod
    def initialize(cls):
        opts = Options()
        if opts.debug:
            cls._log_level = logging.DEBUG

        log_dir = opts.output_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{opts.file_name_base}_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}{opts.custom_desc}.log"
        cls._file_handler = logging.FileHandler(log_path)

        logging.basicConfig(level=cls._log_level,
                    handlers=[
                        cls._stream_handler,
                        cls._file_handler
                    ],
                    format=cls._default_log_format)
        
        cls.root = logging.getLogger()
        cls.operations = logging.getLogger("first_level.operations")
        cls.utils = logging.getLogger("first_level.utils")
        


        
def set_configs(args):

    Options(args) 
    loggers.initialize()
