import sys
import logging
from pathlib import Path
import json
from subprocess import Popen, PIPE
import shlex
from types import SimpleNamespace
import nibabel as nib
import nilearn.masking as nmask
from nilearn.image import smooth_img
import os
import os.path as op
import numpy as np

logger = logging.getLogger(__name__)

default_log_format = "%(levelname)s:%(asctime)s:%(module)s: %(message)s"

flags = SimpleNamespace(debug=False, longitudinal=False, infant=False)

bids_entities = ["sub", "ses", "sample", "task", "tracksys",
                 "acq", "ce", "trc", "stain", "rec", "dir",
                 "run", "mod", "echo", "flip", "inv", "mt",
                 "part", "proc", "hemi", "space", "split", "recording",
                 "chunk", "seg", "res", "den", "label", "desc"]

cifti_files = [
    ".dtseries.nii",
    ".ptseries.nii",
    ".dscalar.nii",
    ".pscalar.nii"
]


def logger_exception_hook(exctype, value, traceback):
    sys.__excepthook__(exctype, value, traceback)
    logger.critical(f'Uncaught exception: {exctype.__name__} - {value}')
    while traceback:
        filename = traceback.tb_frame.f_code.co_filename
        name = traceback.tb_frame.f_code.co_name
        line_no = traceback.tb_lineno
        traceback = traceback.tb_next
        if traceback:
            logger.critical(f"-- File {filename} line {line_no}, in {name} ")

    # Where the exception occured
    logger.exception(f"File {filename} line {line_no}, in {name}", exc_info=(exctype, value, traceback))


sys.excepthook = logger_exception_hook


def takes_arguments(decorator):
    """
    A meta-decorator to use on decorators that take in other
    arguments than just the function they are applied to
    """
    def wrapper(*args, **kwargs):
        def replacement(func):
            return decorator(func, *args, **kwargs)
        return replacement
    return wrapper


def debug_logging(func):
    """
    A decorator function that debug logs a function call
    and the arguments used in the call. Can log with a 
    specified logger or this module's logger if one is 
    not supplied

    :param func: function this decorator is applied to
    :type func: function
    :param this_logger: a logger to use for logging the function call
    :type this_logger: logging.Logger object
    :return: a function that is the input function wrapped by this decorator
    :rtype: function
    """
    def inner(*args, **kwargs):
        log_linebreak()
        logger.debug(
            f"calling - {func.__module__}:{func.__name__}({', '.join([str(a) for a in args] + [f'{k}={v}' for k,v in kwargs.items()])})\n"
        )
        return func(*args, **kwargs)
    return inner


def exit_program_early(msg:str,
                       exit_func=None):
    """
    Exit the program while printing parameter 'msg'. If an exit
    function is sepcified, it will be called before the program
    exits

    :param msg: error message to display.
    :type msg: str
    :param exit_func: function to be called before program exit (no required arguments)
    :type exit_func: function

    """
    log_linebreak()
    logger.error(f"---[ERROR]: {msg} \nExiting the program now...\n")
    if exit_func and callable(exit_func):
        exit_func()
    sys.exit(1)


def prompt_user_continue(msg:str) -> bool:
    """
    Prompt the user to continue with a custom message.

    :param msg: prompt message to display.
    :type msg: str

    """
    prompt_msg = f"{msg} \n\t---(press 'y' for yes, other input will mean no)"
    user_continue = input(prompt_msg + "\n")
    ans = (user_continue.lower() == "y")
    logger.debug(f"User Prompt: {prompt_msg}")
    logger.debug(f"User Response:  {user_continue} ({ans})")
    return ans


def extract_options(option_chain:list) -> dict:
    val = None
    opts = dict()
    key = None
    if len(option_chain) < 1:
        return opts

    if not (isinstance(option_chain[0], str) and option_chain[0].startswith("-")):
        exit_program_early(f"cannot parse option chain: {option_chain}")

    for o in option_chain:
        if isinstance(o, str) and o.startswith("-"):
            if key:
                opts[key] = val if val else True
            key = o.lstrip("-")
            val = None
        elif key:
            if val is None:
                val = o
            elif isinstance(val, list):
                val.append(o)
            else:
                val = [val, o]
    else:
        if key:
            opts[key] = val if val else True
    return opts


def make_option(value,
                key: str = None,
                delimeter: str = " ",
                convert_underscore: bool = False):
    """
    Generate a string, representing an option that gets fed into a subprocess or script.

    For example, if a key is 'option' and its value is True, the option string it will generate would be:

        --option

    If value is equal to some string 'value', then the string would be:

        --option value

    If value is a list of strings:

        --option value1 value2 ... valuen
    :param value: Value to pass in along with the 'key' param.
    :type value: any
    :param key: Name of option, without any hyphen at the beginning.
    :type key: str
    :param delimeter: character to separate the key and the value in the option string. Default is a space.
    :type delimeter: str
    :param convert_underscore: flag to indicate that underscores should be replaced with '-'
    :type convert_underscore: bool
    :return: String to pass as an option into a subprocess call.
    :rtype: str
    """
    second_part = None
    if key and convert_underscore:
        key = key.replace("_", "-")
    if not value:
        return ""
    elif type(value) == bool and value:
        second_part = " "
    elif type(value) == list:
        second_part = f"{delimeter}{' '.join([str(v) for v in value])}"
    else:
        second_part = f"{delimeter}{str(value)}"
    return f"--{key}{second_part}" if key else second_part


def add_file_handler(this_logger: logging.Logger,
                     log_path: Path,
                     format_str: str = default_log_format):
    """
    Adds a file handler with the input file path and input message formatting to the input logger.
    The default message formatting in defined at the top of this file.

    :param this_logger: A logger object to add the file handler to
    :type this_logger: logging.Logger object
    :param log_path: A path to the output file for the file handler
    :type log_path: pathlib.Path
    :param format_str: a string representing the desired message formatting for the file_handler. (See logging.Formatter class for more detail)
    :type format_str: str
    """
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(format_str))
    this_logger.addHandler(file_handler)


def prepare_subprocess_logging(this_logger,
                               stop=False):
    """
    Prepares a logger for piping subprocess outputs to its handlers. This function removes 
    newlines from all of the logger's handlers and simplifies the message format to cleaning 
    display subprocess outputs. If 'stop' is True, the logger handlers are restored to defaults

    :param this_logger: A logger object to change formatting for
    :type this_logger: logging.Logger object
    :param stop: A flag to indicate if subprocessing logging should end
    :type stop: bool
    """
    if stop:
        while this_logger != None:
            for h in this_logger.handlers:
                h.setFormatter(logging.Formatter(default_log_format))
                h.terminator = "\n"
            this_logger = this_logger.parent
    else:
        while this_logger != None:
            for h in this_logger.handlers:
                h.setFormatter(logging.Formatter("%(message)s"))
                h.terminator = ""
            this_logger = this_logger.parent


@debug_logging
def run_subprocess(cmd: str, title: str):
    split_cmd = shlex.split(cmd)
    logger.info(f"running '{title}' with the following command: \n{' '.join(split_cmd)}\n")
    prepare_subprocess_logging(logger)
    with Popen(split_cmd, stdout=PIPE, stderr=PIPE) as p:
        while p.poll() == None:
            for line in p.stdout:
                logger.info(line.decode("utf-8", "ignore"))
            for line in p.stderr:
                logger.info(line.decode("utf-8", "ignore"))
        prepare_subprocess_logging(logger, stop=True)
        p.kill()
        if p.poll() != 0:
            raise RuntimeError(f"process -'{title}'- has ended with a non-zero exit code.")


def log_linebreak():
    """
    Logs a single blank line using this module's logger.
    Changes the formatter for each handler to be a empty, logs
    a single blank line, then changes each formatter back to 
    the default format
    """
    traverse_logger = logger
    while traverse_logger != None:
        for h in traverse_logger.handlers:
            h.setFormatter(logging.Formatter(""))
        traverse_logger = traverse_logger.parent
    logger.info("")
    traverse_logger = logger
    while traverse_logger != None:
        for h in traverse_logger.handlers:
            h.setFormatter(logging.Formatter(default_log_format))
        traverse_logger = traverse_logger.parent


@debug_logging
def export_args_to_file(args,
                        argument_group,
                        file_path: Path,
                        extra_args:dict = None):
    """
    Takes the arguments in the argument group, and exports their names and values in the 'args'
    namespace to a file specified at 'file_path'. The input 'file_path' can either be a txt
    file or a json file.

    :param args: an argument namespace to pull input values from
    :type args: argparse.Namespace
    :param argument_group: The argument group representing the subset of inputs to save to a file
    :type argument_group: argparse._ArgumentGroup
    :param file_path: a path to a file where the arguments should be saved
    :type file_path: pathlib.Path
    """

    all_opts = dict(args._get_kwargs())
    opts_to_save = dict()
    for a in argument_group._group_actions:
        if a.dest in all_opts and all_opts[a.dest]:
            if type(all_opts[a.dest]) == bool:
                opts_to_save[a.option_strings[0]] = ""
            elif isinstance(all_opts[a.dest], Path):
                opts_to_save[a.option_strings[0]] = str(all_opts[a.dest].resolve())
            elif isinstance(all_opts[a.dest], list):
                opts_to_save[a.option_strings[0]] = [v if isinstance(v, int) or isinstance(v, str) else str(v) for v in all_opts[a.dest]]
            else:
                opts_to_save[a.option_strings[0]] = all_opts[a.dest]

    if extra_args:
        for key, val in extra_args.items():
            opt_key = make_option(True, key).strip()
            if val:
                if isinstance(val, bool):
                    opts_to_save[opt_key] = ""
                elif isinstance(val, Path):
                    opts_to_save[opt_key] = str(val.resolve())
                else:
                    opts_to_save[opt_key] = val

    with open(file_path, "w") as f:
        if file_path.suffix == ".json":
            f.write(json.dumps(opts_to_save, indent=4))
        else:
            for k,v in opts_to_save.items():
                f.write(f"{k}{make_option(value=v)}\n")


def is_cifti_file(file: str | Path) -> str | None:
    if isinstance(file, Path):
        file = str(file)
    suffix = [cf for cf in cifti_files if file.endswith(cf)]
    return suffix[0] if len(suffix) > 0 else None


def is_nifti_file(file: str | Path) -> str | None:
    if isinstance(file, Path):
        file = str(file)
    not_cifti = is_cifti_file(file) is None
    suffix = [nf for nf in [".nii.gz", ".nii"] if file.endswith(nf)]
    return suffix[0] if (len(suffix) > 0) and not_cifti else None


@debug_logging
def load_data(func_file: str | Path,
              brain_mask: str = None,
              need_tr: bool = False,
              fwhm: float = None) -> np.ndarray:
    func_file = str(func_file)
    if op.islink(func_file):
        func_file = os.readlink(func_file)

    if brain_mask is not None and op.islink(brain_mask):
        brain_mask = os.readlink(brain_mask)

    tr = None
    if need_tr:
        sidecar_file = func_file.split(".")[0] + ".json"
        assert op.isfile(sidecar_file), f"Cannot find the .json sidecar file for bold run: {func_file}"
        with open(sidecar_file, "r") as f:
            jd = json.load(f)
            tr = jd["RepetitionTime"]

    if is_cifti_file(func_file):
        img = nib.load(func_file)
        return (img.get_fdata(), tr, img.header)
    elif is_nifti_file(func_file):
        if brain_mask:
            img = nib.load(func_file)
            if fwhm is not None:
                img = smooth_img(img, fwhm)
            return (nmask.apply_mask(img, brain_mask), tr, None)
        else:
            raise Exception("Volumetric data must also have an accompanying brain mask")
            # return None


def parcellate_dtseries(dtseries_path: Path,
                        parc_dlabel_path: Path) -> Path:

    ptseries_path = Path(str(dtseries_path.resolve()).replace("dtseries", "ptseries"))
    parcellate_cmd = f"""wb_command -cifti-parcellate {dtseries_path.resolve()}
                            {parc_dlabel_path.resolve()} COLUMN {ptseries_path.resolve()}"""
    title = "wb_command"
    try:
        run_subprocess(parcellate_cmd, title="wb_command")
    except RuntimeError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early(f"Program '{title}' has run into an error.")

    return ptseries_path
