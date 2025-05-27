import argparse
from pathlib import Path
import json
import yaml
from typing import List, Dict, Union, Optional


from pathlib import Path
from typing import Any, Dict

def substitute_paths(obj: Any,
                     substitutions: Dict[str, str],
                     verbose: bool = False) -> Any:
    """
    Recursively walk a data structure (dicts / lists / scalars) and
    replace every substring that matches one of the keys in *substitutions*
    with the corresponding value.

    Parameters
    ----------
    obj : Any
        The object to process: dict, list, str, Path, or a scalar.
    substitutions : Dict[str, str]
        Mapping old_path → new_path.
    verbose : bool, optional
        If True, print every replacement that is performed.

    Returns
    -------
    Any
        A deep-copied structure with paths substituted.
    """
    # sort by length so the more specific (longer) paths fire first
    ordered_items = sorted(substitutions.items(), key=lambda kv: -len(kv[0]))

    if isinstance(obj, dict):
        return {k: substitute_paths(v, substitutions, verbose) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [substitute_paths(item, substitutions, verbose) for item in obj]

    elif isinstance(obj, (str, Path)):
        text = str(obj)
        for old, new in ordered_items:
            if old in text:
                if verbose:
                    print(f"Substituting {old!r} → {new!r} in {text!r}")
                text = text.replace(old, new)
        # keep original type (str or Path) on output
        return obj.__class__(text) if isinstance(obj, Path) else text

    else:
        # int, float, bool, None … leave untouched
        return obj



def load_config(default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/train_landmarks_01.yaml",
                debugging_in_jupyter_nb=False, silencium=False, return_config_name=False, 
                default_path_substitution_config=None):

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Read in the filename of the config file (JSON or YAML).")
        parser.add_argument('--config', type=str,
                            help=f"Path to the config file. Supported formats: .json, .yaml, .yml\n default = ({default_config})",
                            default=default_config  # for debugging
                            )
        
        parser.add_argument('--path_substitution_config', type=str,
                            help=f"Path to the config file used for changing base_paths\n default = ({default_path_substitution_config})",
                            default=default_path_substitution_config
                            )
                
        
        parser.add_argument(
            '--debugging',
            action='store_true',
            help="Enable debugging mode. Default is False."
        )
        return parser.parse_args()

    if debugging_in_jupyter_nb:
        config_file = Path(default_config)
        if not silencium:
            print("Debugging in Jupyter notebook")
    else:
        args = parse_args()
        config_file = Path(args.config)

    # Check if the file exists
    if not config_file.exists():
        raise FileNotFoundError(f"The file {config_file} does not exist.")

    # Determine file type based on extension and load accordingly
    if config_file.suffix.lower() == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    elif config_file.suffix.lower() in ['.yaml', '.yml']:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file extension '{config_file.suffix}'. Supported extensions are .json, .yaml, .yml.")

    if args.debugging:
        config["debugging"] = True
        print("Debugging mode is enabled with --debugging!! Config will be modified accordingly.")

    if args.path_substitution_config: 
        path_substitution_file = Path(args.path_substitution_config)
        if not path_substitution_file.exists():
            raise FileNotFoundError(f"The path substitution config file {path_substitution_file} does not exist.")
        with open(path_substitution_file, 'r') as f:
            path_substitution_dct = yaml.safe_load(f)
        config = substitute_paths(config, path_substitution_dct)
        print(f"Substituted paths in config using {path_substitution_file}")

    if not silencium:
        print("CONFIG:", config_file)
        from pprint import pprint
        pprint(config)

    if return_config_name:
        return config, config_file
    else: 
        return config