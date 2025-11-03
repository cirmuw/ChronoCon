import os
import torch
from typing import List, Dict, Tuple, Union, Set
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
import numpy as np
import pkg_resources
import git
import argparse
from pprint import pprint
import json
import yaml
import pydoc



def get_package_rootdir(package, level=1):
    # Note: this will only work with level=1 if the git repo has the standard structure
    path = Path(os.path.abspath(package.__file__))
    return path.parents[level]


def get_git_infos(repo_path):
    git_folder = Path(repo_path,'.git')
    if not git_folder.exists():
        return {"message": f"The given path {repo_path} is not a git repository."}
    
    try:
        git_repo = git.Repo(repo_path)
        is_dirty = git_repo.is_dirty()
        commit_message = git_repo.head.commit.message
        # Use hexsha directly - works for both normal and detached HEAD state
        commit = git_repo.head.commit.hexsha
        
        return dict(commit=commit, commit_message=commit_message, dirty=is_dirty)
    
    except FileNotFoundError as e:
        return {
            "message": f"Git information could not be retrieved: {str(e)}. This may occur in detached HEAD state or corrupted git repository.",
            "error": str(e)
        }
    except Exception as e:
        return {
            "message": f"Error retrieving git information: {str(e)}",
            "error": str(e)
        }
    

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='.'):
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_nested = result
        for part in parts[:-1]:
            if part not in d_nested:
                d_nested[part] = {}
            d_nested = d_nested[part]
        d_nested[parts[-1]] = value
    return result

# For logging with mlflow
def log10_params_dct(config):    
    flat_cfg = flatten_dict(config)
    log_params = {}
    for k in flat_cfg:
        v = flat_cfg[k]
        if isinstance(v, float) and v > 0:
            k = f"{k}  log10"
            log_params[k] = np.log10(v)
    return log_params


def import_kwargs_with_pydoc(kwargs: dict, kwargs_req_import : list = [], raise_if_not_found=True):
    _kwargs = dict(**kwargs)
    for ri in kwargs_req_import:
        if ri not in _kwargs.keys():
            print(f"{ri} not imported since it is not in the given dict")
        if _kwargs[ri] is not None:
            tmp = _kwargs[ri]
            _kwargs[ri] = pydoc.locate(_kwargs[ri])
            if raise_if_not_found and _kwargs[ri] == None: 
                raise ImportError(f"Import with pydoc failed {ri}: {tmp} not found")
    return _kwargs

def pydoc_locate_targets(import_targets : List[str], chill=False):
    out = [pydoc.locate(x) for x in import_targets]
    if not chill: 
        for i, o in enumerate(out): 
            if o == None:
                raise ImportError(f"Could not locate {import_targets[i]}")
    return out


def convert_list_like_dict_to_list(list_like_dict):
    if isinstance(list_like_dict, list):
        return list_like_dict
    elif isinstance(list_like_dict, dict):
            
        # Convert keys to integers and sort them to maintain order
        sorted_keys = sorted(list_like_dict.keys(), key=int)
        
        # Create a list by accessing values corresponding to sorted keys
        result_list = [list_like_dict[key] for key in sorted_keys]
    else: 
        raise TypeError(f"can not convert type: {type(list_like_dict)}. Expected list or dict")    
    
    return result_list


# For reading params from yml
def model_parameter_imports_(model_params: Dict[str, str], 
            model_kw_requires_import: List[str], 
            model_dct_keys_to_convert_to_lists: List[str]):
    
    dct = model_params.copy()
    
    keys_for_imports = [k for k in dct.keys() if k in model_kw_requires_import]
    values_for_imports = [model_params[k] for k in keys_for_imports]
    
    imports = pydoc_locate_targets(values_for_imports, chill=False)
    for key, imp in zip(keys_for_imports, imports):
        dct[key] = imp
                    
    for k in model_dct_keys_to_convert_to_lists:
        tmp_list = convert_list_like_dict_to_list(dct[k])
        dct[k] = tmp_list
        
    # convert    dropout_op_kwargs.p = 0  to {"p": 0.0} 
    dct = unflatten_dict(dct)

    return dct


def get_optional_config_parameter(config: dict, key: str, default_value=None, del_from_config=False, verbose=True):
    if key in config:
        v = config[key]
        if del_from_config:
            del config[key]
    else:
        v = default_value
        if verbose:
            print(f"Key '{key}' not found in config. Using default value: '{v}'")
    
    return v


def package_infos(package, level=1):
    package_root = get_package_rootdir(package, level=level)
    infos = get_git_infos(package_root)
    version = pkg_resources.get_distribution(package.__name__).version
    r = dict(name = package.__name__, 
             version=version, 
             package_root=str(package_root))
    return {**r, **infos}


def file_pairs_to_replacement_dicts(pairs: List[List[str]]):
    dAB = {}
    dBA = {}
    for a,b in pairs: 
        dAB[a] = b
        dBA[b] = a
    return dAB, dBA 

def reroot_filepath(p: str, roots: dict, chill=False):
    for root in roots.keys():
        if root in p: 
            return p.replace(root, roots[root])
    if not chill:
        raise ValueError(f"Cound not find in {p} any of the  {roots.keys()}")
    else: 
        return p



def datestr_to_years_since_2000(date_strs, avg_year_days: float = 365.2425, 
                                np_dtype=np.float32#, torch_device: str = "cpu"
                                ):
    """
    Convert ['YYYYMMDD', ...] -> numpy float array of years since 2000-01-01,
    then to a torch.float32 tensor on the chosen device.
    """
    # 1) Normalize to ISO strings 'YYYY-MM-DD' (vectorizable, no loops in the math)
    iso = np.array([f"{s[:4]}-{s[4:6]}-{s[6:8]}" for s in date_strs], dtype="U10")

    # 2) NumPy datetime arithmetic in days
    dates = iso.astype("datetime64[D]")
    base  = np.datetime64("2000-01-01", "D")
    days  = (dates - base).astype("timedelta64[D]").astype(np.int64)

    # 3) Convert to (fractional) years using mean Gregorian year length
    years_np = (days.astype(np.float64) / avg_year_days).astype(np_dtype)

    # 4) NumPy -> Torch tensor
    # years_t = torch.from_numpy(years_np).to(torch_device)
    return years_np#, years_t

# years_np = datestr_to_years_since_2000(batch["date_str"])
# years_t = torch.from_numpy(years_np).to(torch_device)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
