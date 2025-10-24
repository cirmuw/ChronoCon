import argparse
from pathlib import Path
import json
import yaml
from typing import List, Dict, Union, Optional
from copy import deepcopy

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


import re
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict

def auto_replace_strings(
    obj: Any,
    auto_replace: Dict[str, Any],
    token_prefix: str = "$$__",
    token_suffix: str = "__$$",
    verbose: bool = False,
    _preserve_type: bool = True,
) -> Any:
    """
    Recursively replace tokens $$__KEY__$$ in strings/Paths within *obj* using values
    from *auto_replace*[KEY]. If KEY is not present, the token is left as-is.

    Parameters
    ----------
    obj : Any
        Config structure (dict/list/str/Path/scalars).
    auto_replace : Dict[str, Any]
        Mapping of keys to replacement values. Values are cast to str.
    token_prefix, token_suffix : str
        Delimiters around the KEY token.
    verbose : bool
        If True, print every replacement performed.
    _preserve_type : bool
        Internal flag to keep Path vs str types on return.

    Returns
    -------
    Any
        A deep-copied structure with replacements applied.
    """
    # Compile regex like r"\$\$__([A-Za-z0-9_]+)__\$\$"
    # Escape the delimiters to be safe
    patt = re.compile(
        re.escape(token_prefix) + r"([A-Za-z0-9_]+)" + re.escape(token_suffix)
    )

    # Normalize replacement values to strings (once)
    replace_str = {k: str(v) for k, v in (auto_replace or {}).items()}

    def _replace_in_text(text: str) -> str:
        def _sub_fn(m: re.Match) -> str:
            key = m.group(1)
            if key in replace_str:
                new = replace_str[key]
                if verbose:
                    print(f"Replacing token {m.group(0)!r} → {new!r} in {text!r}")
                return new
            else:
                # leave unknown token unchanged
                if verbose:
                    print(f"Leaving unknown token {m.group(0)!r} unchanged in {text!r}")
                return m.group(0)

        # Repeat until stable in case replacements themselves contain further tokens
        # (bounded by a small iteration cap to avoid accidental loops)
        max_iters = 5
        prev = text
        for _ in range(max_iters):
            new = patt.sub(_sub_fn, prev)
            if new == prev:
                break
            prev = new
        return prev

    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            # Only replace in values (not keys)
            return {k: _walk(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [_walk(i) for i in x]
        elif isinstance(x, tuple):
            return tuple(_walk(i) for i in x)
        elif isinstance(x, Path):
            new_text = _replace_in_text(str(x))
            return Path(new_text) if _preserve_type else new_text
        elif isinstance(x, str):
            return _replace_in_text(x)
        else:
            # numbers, bools, None, etc.
            return x

    return _walk(deepcopy(obj))




def load_config(default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/train_landmarks_01.yaml",
                debugging_in_jupyter_nb=False, silencium=False, return_config_name=False, 
                return_originals = False, 
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

    originals = {}

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
    originals["original_config"] = deepcopy(config)

    if not debugging_in_jupyter_nb and args.debugging:
        config["debugging"] = True
        print("Debugging mode is enabled with --debugging!! Config will be modified accordingly.")

    AUTO_REPLACE_DCT = config.get("AUTO_REPLACE")
    if AUTO_REPLACE_DCT is not None: 
        config = auto_replace_strings(config, AUTO_REPLACE_DCT, verbose= (not silencium))



    if not debugging_in_jupyter_nb and args.path_substitution_config: 
        path_substitution_file = Path(args.path_substitution_config)
        if not path_substitution_file.exists():
            raise FileNotFoundError(f"The path substitution config file {path_substitution_file} does not exist.")
        with open(path_substitution_file, 'r') as f:
            path_substitution_dct = yaml.safe_load(f)
            originals["original_path_substitution_dct"] = deepcopy(path_substitution_dct)
        config = substitute_paths(config, path_substitution_dct)
        print(f"Substituted paths in config using {path_substitution_file}")

    if not silencium:
        print("CONFIG:", config_file)
        from pprint import pprint
        pprint(config)

    if return_config_name:
        if return_originals: 
            return config, config_file, originals
        else: 
            return config, config_file
    else: 
        if return_originals: 
            return config, originals
        else: 
            return config
    