import sys
import logging
from pathlib import Path
import json
from types import SimpleNamespace

logger = logging.getLogger(__name__)

default_log_format = "%(levelname)s:%(asctime)s:%(module)s: %(message)s"

flags = SimpleNamespace(debug = False)

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
    user_continue = input(prompt_msg+"\n")
    ans = (user_continue.lower() == "y")
    logger.debug(f"User Prompt: {prompt_msg}")
    logger.debug(f"User Response:  {user_continue} ({ans})")
    return ans


def make_option(value, 
                key: str=None, 
                delimeter: str=" ", 
                convert_underscore: bool=False):
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
                     format_str: str=default_log_format):
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


def export_args_to_file(args, 
                        argument_group, 
                        file_path: Path):
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
                continue
            elif isinstance(all_opts[a.dest], Path):
                opts_to_save[a.option_strings[0]] = str(all_opts[a.dest])
                continue
            opts_to_save[a.option_strings[0]] = all_opts[a.dest]
    with open(file_path, "w") as f:
        if file_path.suffix == ".json":
            f.write(json.dumps(opts_to_save, indent=4))
        else:
            for k,v in opts_to_save.items():
                f.write(f"{k}{make_option(value=v)}\n")
