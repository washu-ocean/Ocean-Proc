import argparse
from pathlib import Path
from ..oceanparse import OceanParser
import logging
from ..utils import flags, log_linebreak, export_args_to_file, exit_program_early
import bids

VERSION = "1.1.4"
logger = logging.getLogger("parser")


def _build_parser():

    # Build out some useful argument types
    def ExistingPath(path):
        p = Path(path)
        if not p.exists():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing path"
            )
        return p.resolve()

    def ExistingDir(path):
        p = ExistingPath(path)
        if not p.is_dir():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing directory"
            )
        return p

    def ExistingFile(path):
        p = ExistingPath(path)
        if not p.is_file():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing file"
            )
        return p

    def ParentExists(path):
        p = Path(path).resolve()
        if not p.parent.exists():
            raise argparse.ArgumentTypeError(
                f"the parent directory for path string <{path}> does not exist"
            )
        return p

    def PositiveVal(val, dtype=int):
        valid = True
        out = None
        if isinstance(val, list):
            try:
                out = [dtype(v) for v in val]
                for v in out:
                    if v <= 0:
                        valid = False
                        break
            except:
                valid = False
        elif isinstance(val, str):
            try:
                out = dtype(val)
                valid = (out >= 0)
            except:
                valid = False
        else:
            valid = False
        if not valid:
            raise argparse.ArgumentTypeError(
                f"The value(s) supplied must be numerical and greater than zero: {val}"
            )
        return out

    def PositiveInt(val):
        return PositiveVal(val, int)

    def PositiveFloat(val):
        return PositiveVal(val, float)

    # Create the argument parser
    parser = OceanParser(
        prog="oceanfla",
        description="Ocean Labs first level analysis",
        fromfile_prefix_chars="@",
        epilog="An arguments file can be accepted with @FILEPATH"
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")

    session_arguments = parser.add_argument_group("Session Specific")

    session_arguments.add_argument("--subject", "-su", required=True,
                                   help="The subject ID")

    session_arguments.add_argument("--session", "-se", required=True,
                                   help="The session ID")

    session_arguments.add_argument("--events_long", "-el", type=ExistingDir, nargs="?", const=lambda a: a.derivs_dir / a.preproc_subfolder,
                                   help="""Path to the directory containing long formatted event files to use. 
                                   Default is the derivatives directory containing preprocessed outputs.""")

    session_arguments.add_argument("--export_args", "-ea", type=ParentExists,
                                   help="Path to a file to save the current arguments.")

    session_arguments.add_argument("--force_overwrite", action="store_true",
                                   help="Use this flag to force oceanfla to proceed when conflicting task outputs are present in the output directory")

    session_arguments.add_argument("--debug", action="store_true",
                                   help="Use this flag to save intermediate outputs for a chance to debug inputs")


    config_arguments = parser.add_argument_group(
        "Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")

    config_arguments.add_argument("--task", "-t", nargs="+", required=True,
                                  help="The name of the task to analyze.")

    # config_arguments.add_argument("--bold_file_type", "-ft", required=True,
    #                               help="The file type of the functional runs to use.")

    config_arguments.add_argument("--brain_mask", "-bm", type=ExistingFile,
                                  help="If the bold file type is volumetric data, a brain mask must also be supplied.")

    config_arguments.add_argument("--func_space", default="fsLR",
                                  help="Space that the preprocessed data should be in (for example, 'T2w', 'MNIInfant', etc.)")

    config_arguments.add_argument("--fwhm", type=PositiveFloat,
                                  help="FWHM smoothing kernel, in mm (only applies to CIFTI data)")

    config_arguments.add_argument("--derivs_dir", "-d", type=ExistingDir, required=True,
                                  help="Path to the BIDS formatted derivatives directory containing processed outputs.")

    config_arguments.add_argument("--preproc_subfolder", "-pd", type=str, default="fmriprep",
                                  help="Name of the subfolder in the derivatives directory containing the preprocessed bold data. Default is 'fmriprep'")

    config_arguments.add_argument("--raw_bids", "-r", type=ExistingDir, required=True,
                                  help="Path to the BIDS formatted raw data directory for this dataset.")

    config_arguments.add_argument("--derivs_subfolder", "-ds", default="first_level",
                                  help="The name of the subfolder in the derivatives directory where bids style outputs should be stored. The default is 'first_level'.")

    config_arguments.add_argument("--output_dir", "-o", type=ExistingDir,
                                  help="Alternate Path to a directory to store the results of this analysis. Default is '[derivs_dir]/first_level/'")

    config_arguments.add_argument("--custom_desc", "-cd",
                                  help="A custom description to add in the file name of every output file.")

    config_arguments.add_argument("--fir", "-ff", type=PositiveInt,
                                  help="The number of frames to use in an FIR model.")

    config_arguments.add_argument("--fir_vars", nargs="*",
                                  help="""A list of the task regressors to apply this FIR model to. The default is to apply it to all regressors if no 
                                  value is specified. A list must be specified if both types of models are being used""")

    config_arguments.add_argument("--hrf", nargs=2, type=PositiveInt, metavar=("PEAK", "UNDER"),
                                  help="""Two values to describe the hrf function that will be convolved with the task events. 
                                  The first value is the time to the peak, and the second is the undershoot duration. Both in units of seconds.""")

    config_arguments.add_argument("--hrf_vars", nargs="*",
                                  help="""A list of the task regressors to apply this HRF model to. The default is to apply it to all regressors if no 
                                  value is specifed. A list must be specified if both types of models are being used.""")

    config_arguments.add_argument("--custom_hrf", "-ch", type=ExistingFile,
                                  help="The path to a txt file containing the timeseries for a custom hrf to use instead of the double gamma hrf")

    config_arguments.add_argument("--unmodeled", "-um", nargs="*",
                                  help="""A list of the task regressors to leave unmodeled, but still included in the final design matrix. These are 
                                  typically continuous variables that need not be modeled with hrf or fir, but any of the task regressors can be included.""")

    config_arguments.add_argument("--start_censoring", "-sc", type=PositiveInt, default=0,
                                  help="The number of frames to censor out at the beginning of each run. Typically used to censor scanner equilibrium time. Default is 0")

    config_arguments.add_argument("--confounds", "-c", nargs="+", default=[],
                                  help="A list of confounds to include from each confound timeseries tsv file.")

    config_arguments.add_argument("--fd_threshold", "-fd", type=PositiveFloat, default=0.9,
                                  help="The framewise displacement threshold used when censoring high-motion frames")

    config_arguments.add_argument("--minimum_unmasked_neighbors", "-mun", type=PositiveInt, default=None,
                                  help="Minimum number of contiguous unmasked frames on either side of a given frame that's required to be under the fd_threshold; any unmasked frame without the required number of neighbors will be masked.")

    config_arguments.add_argument("--tmask", action=argparse.BooleanOptionalAction,
                                  help="Flag to indicate that tmask files, if found with the preprocessed outputs, should be used. Tmask files will override framewise displacement threshold censoring if applicable.")

    config_arguments.add_argument("--repetition_time", "-tr", type=PositiveFloat,
                                  help="Repetition time of the function runs in seconds. If it is not supplied, an attempt will be made to read it from the JSON sidecar file.")

    config_arguments.add_argument("--detrend_data", "-dd", action="store_true",
                                  help="""Flag to demean and detrend the data before modeling. The default is to include 
                                  a mean and trend line into the nuisance matrix instead.""")
    
    config_arguments.add_argument("--percent_change", "-pc", action="store_true",
                                  help="""Flag to convert data to percent signal change.""")

    config_arguments.add_argument("--no_global_mean", action="store_true",
                                  help="Flag to indicate that you do not want to include a global mean into the model.")

    high_motion_params = config_arguments.add_mutually_exclusive_group()
    high_motion_params.add_argument("--spike_regression", "-sr", action="store_true",
                                    help="Flag to indicate that framewise displacement spike regression should be included in the nuisance matrix.")

    high_motion_params.add_argument("--fd_censoring", "-fc", action="store_true",
                                    help="Flag to indicate that frames above the framewise displacement threshold should be censored before the GLM.")

    config_arguments.add_argument("--run_exclusion_threshold", "-re", type=PositiveInt,
                                  help="The percent of frames a run must retain after high motion censoring to be included in the fine GLM. Only has effect when '--fd_censoring' is active.")

    config_arguments.add_argument("--nuisance_regression", "-nr", nargs="*", default=[],
                                  help="""List of variables to include in nuisance regression before the performing the GLM for event-related activation. If no values are specified then
                                  all nuisance/confound variables will be included""")

    config_arguments.add_argument("--nuisance_fd", "-nf", type=PositiveFloat,
                                  help="The framewise displacement threshold used when censoring frames for nuisance regression.")

    config_arguments.add_argument("--highpass", "-hp", type=PositiveFloat, nargs="?", const=0.008,
                                  help="""The high pass cutoff frequency for signal filtering. Frequencies below this value (Hz) will be filtered out. If the argument 
                                  is supplied but no value is given, then the value will default to 0.008 Hz""")

    config_arguments.add_argument("--lowpass", "-lp", type=PositiveFloat, nargs="?", const=0.1,
                                  help="""The low pass cutoff frequency for signal filtering. Frequencies above this value (Hz) will be filtered out. If the argument 
                                  is supplied but no value is given, then the value will default to 0.1 Hz""")

    config_arguments.add_argument("--filter_padtype", default="mean",
                                  choices=["odd", "even", "zero", "none", "mean", "edge"],
                                  help="Type of padding to use for low-, high-, or band-pass filter, if one is applied. The default is 'mean'")

    config_arguments.add_argument("--filter_padlen", type=PositiveInt, default=50,
                                  help="Length of padding to add to the beginning and end of BOLD run before applying butterworth filter. The default is 50")

    config_arguments.add_argument("--volterra_lag", "-vl", nargs="?", const=2, type=PositiveInt,
                                  help="""The amount of frames to lag for a volterra expansion. If no value is specified 
                                  the default of 2 will be used. Must be specifed with the '--volterra_columns' option.""")

    config_arguments.add_argument("--volterra_columns", "-vc", nargs="+", default=[],
                                  help="The confound columns to include in the expansion. Must be specifed with the '--volterra_lag' option.")

    config_arguments.add_argument("--parcellate", "-parc", type=ExistingFile,
                                  help="Path to a dlabel file to use for parcellation of a dtseries")

    config_arguments.add_argument("--stdscale_glm", choices=["runlevel", "seslevel", "both", "none"], default="seslevel",
                                  help="Option to standard scale concatenated timeseries before running final GLM (after masking & nuisance regression)")


    return (parser, config_arguments)


# Function to parse the command line arguments and
#   validate them before they become global options
def parse_args():

    parser, config_arguments = _build_parser()
    args = parser.parse_args()

    # don't allow ambiguity when modeling variables two separate ways
    if args.hrf is not None and args.fir is not None:
        if not args.fir_vars or not args.hrf_vars:
            parser.error(
                "Must specify variables to apply each model to if using both types of models")
    elif args.hrf is None and args.fir is None:
        parser.error(
            "Must include model parameters for at least one of the models, fir or hrf.")

    if args.custom_hrf:
        if not (args.custom_hrf.exists() and args.custom_hrf.suffix == ".txt"):
            parser.error(
                "The 'custom_hrf' argument must be a file of type '.txt' and must exist")

    # if args.bold_file_type[0] != ".":
    #     args.bold_file_type = "." + args.bold_file_type
    # if args.bold_file_type == ".nii" or args.bold_file_type == ".nii.gz":
    #     args.imagetype = "nifti"
    # else:
    #     args.imagetype = "cifti"

    if args.parcellate:
        if (not args.parcellate.exists()) or (not args.parcellate.name.endswith(".dlabel.nii")):
            parser.error(
                "The 'parcellate' argument must be a file of type '.dlabel.nii' and must exist")

    # flags.parcellated = (
    #     args.parcellate or args.bold_file_type == ".ptseries.nii")

    if (args.volterra_lag and not args.volterra_columns) or (not args.volterra_lag and args.volterra_columns):
        parser.error(
            "The options '--volterra_lag' and '--volterra_columns' must be specifed together, or neither of them specified.")

    if callable(args.events_long):
        args.events_long = args.events_long(args)

    # Add bids layouts for both bids directories
    args.preproc_bids = args.derivs_dir / args.preproc_subfolder
    if not args.preproc_bids.exists():
        parser.error(
            f"The preprocessed outputs directory does not exist at path: {args.preproc_dir}")

    args.raw_layout = bids.BIDSLayout(root=args.raw_bids,
                                      database_path=args.raw_bids / ".bids_indexer",
                                      reset_database=True,
                                      validate=False,
                                      absolute_paths=True)
    args.preproc_layout = bids.BIDSLayout(root=args.preproc_bids,
                                          database_path=args.preproc_bids / ".bids_indexer",
                                          reset_database=True,
                                          validate=False,
                                          absolute_paths=True,
                                          is_derivative=True)

  
    # Export the current arguments to a file
    if args.export_args:
        try:
            assert args.export_args.parent.is_dir(), "Argument export path must be a file path in a directory that exists"
            log_linebreak()
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######\n")
            export_args_to_file(args, config_arguments, args.export_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    args.custom_desc = f"-{args.custom_desc}" if args.custom_desc else ""
    args.file_name_base = f"sub-{args.subject}_ses-{args.session}_task-{args.task}"

    if not hasattr(args, "output_dir") or args.output_dir is None:
        args.output_dir = args.derivs_dir / \
            f"{args.derivs_subfolder}/sub-{args.subject}/ses-{args.session}/func"

    '''
    TODO: create singleton options class so arguments are passed to each process
    '''
    from .config import set_configs
    set_configs(args.__dict__)
