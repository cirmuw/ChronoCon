""" For Hyperparameter search. 
A lot of the code originates from 
# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html


Some other parts: 

Goal 1:
------- 
Read paramters for HP search from config of the form: 

  "transform_params": {
    "fixed_parameters": {
      "transform__mask_to_ROI": true, 
      ...
    },
    "optuna_hp_ranges": [
      {"type": "suggest_float", "params": {"name": "transform__blur_prob", "low": 0.0, "high": 0.8}}, 
      ...
    ]
  },


Goal 2: 
-------
Recreate config from trial.params:

  "transform_params": {
    "fixed_parameters": {
      "transform__mask_to_ROI": true, 
      ...
    },
    "optuna_hp_ranges": [
      {"type": "suggest_float", "params": {"name": "transform__blur_prob", "low": 0.0, "high": 0.8}}, 
      ...
    ],
   "optuna_params": {"param_name": value} 
    
  },





"""

from copy import deepcopy
import optuna
import mlflow
import mlflow.pytorch
import mlflow
import ast



def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    


# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")



def replace_none_recursive(data, none_key = "__NonePlaceholder__"):
    """
    Recursively replace all string values equal to '__NonePlaceholder__' with None.
    
    Works for nested dicts and lists.
    """
    if isinstance(data, dict):
        return {k: replace_none_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_none_recursive(v) for v in data]
    elif isinstance(data, str) and data == none_key:
        return None
    else:
        return data




# def generate_optuna_params(trial, config: dict):
#     assert "fixed_parameters" in config.keys()
#     assert "optuna_hp_ranges" in config.keys()

    
#     # assert that there is no overlap between the keys
#     overlap = set(config["fixed_parameters"].keys()) & set([a["params"]["name"] for a in  config["optuna_hp_ranges"]])
#     assert overlap == set(), f"Parameters must be either in 'fixed_parameters' or in 'optuna_hp_ranges', not in both. Check: '{overlap}'"

    
#     trial_params = {}
#     for optuna_hp_range in config["optuna_hp_ranges"]:
#         type_ = optuna_hp_range["type"]
#         name_ = optuna_hp_range["params"]["name"]
#         params_ = optuna_hp_range["params"]
        
#         if type_ == "suggest_categorical":
#             v = trial.suggest_categorical(**params_)
#         elif type_ == "suggest_discrete_uniform":
#             v = trial.suggest_discrete_uniform(**params_)
#         elif type_ == "suggest_float":
#             v = trial.suggest_float(**params_)
#         elif type_ == "suggest_int":
#             v = trial.suggest_int(**params_)
#         elif type_ == "suggest_loguniform":
#             v = trial.suggest_float(**params_, log=True)
#         elif type_ == "suggest_uniform":
#             v = trial.suggest_uniform(**params_)
#         else: 
#             raise ValueError(f"type '{type_}' for optuna hp ranges not known or not implemented yet!")   
        
#         valid_special_treatents = ["literal_eval"]
#         if "special_treatment" in optuna_hp_range.keys():
#             s = optuna_hp_range["special_treatment"]
#             assert s in valid_special_treatents, f"{s} must be one of {valid_special_treatents}. Other choices are not valid." 
#             if s == "literal_eval":
#                 v = ast.literal_eval(v)
#         trial_params[name_] = v
        
#     combined_parameters = {**trial_params, **config["fixed_parameters"]}        

#     return combined_parameters    


def generate_optuna_params(trial, config: dict, prefix: str | None = None):
    """
    Create trial parameters from an optuna block.
    Adds 'prefix' (e.g. 'model.encoder') in front of every Optuna parameter 'name'
    to make names globally unique and avoid dynamic value-space errors.
    """
    assert "fixed_parameters" in config, "Missing 'fixed_parameters'"
    assert "optuna_hp_ranges" in config, "Missing 'optuna_hp_ranges'"

    # Check there is no overlap between final keys:
    # We compare against *prefixed* names, since that's what we'll actually use.
    prefixed_names = []
    for a in config["optuna_hp_ranges"]:
        base = a["params"]["name"]
        full = f"{prefix}.{base}" if prefix else base
        prefixed_names.append(full)

    overlap = set(config["fixed_parameters"].keys()) & set(prefixed_names)
    assert overlap == set(), (
        "Parameters must be either in 'fixed_parameters' or in 'optuna_hp_ranges', not both. "
        f"Overlapping keys after prefixing: {overlap}"
    )

    trial_params = {}
    for optuna_hp_range in config["optuna_hp_ranges"]:
        type_ = optuna_hp_range["type"]
        params_ = deepcopy(optuna_hp_range["params"])

        # Build unique hierarchical name
        base_name = params_["name"]
        full_name = f"{prefix}.{base_name}" if prefix else base_name
        params_["name"] = full_name

        # Make categorical choices deterministic (and hashable) if present
        if type_ == "suggest_categorical" and "choices" in params_:
            params_["choices"] = _stable_choices(params_["choices"])

        # Dispatch
        if type_ == "suggest_categorical":
            v = trial.suggest_categorical(**params_)
        elif type_ == "suggest_discrete_uniform":
            v = trial.suggest_discrete_uniform(**params_)
        elif type_ == "suggest_float":
            v = trial.suggest_float(**params_)
        elif type_ == "suggest_int":
            v = trial.suggest_int(**params_)
        elif type_ == "suggest_loguniform":
            v = trial.suggest_float(**params_, log=True)
        elif type_ == "suggest_uniform":
            v = trial.suggest_uniform(**params_)
        else:
            raise ValueError(f"type '{type_}' for optuna hp ranges not known or not implemented yet!")

        # Optional special handling
        if "special_treatment" in optuna_hp_range:
            s = optuna_hp_range["special_treatment"]
            valid = ["literal_eval"]
            assert s in valid, f"{s} must be one of {valid}. Other choices are not valid."
            if s == "literal_eval":
                v = ast.literal_eval(v)

        trial_params[base_name] = v

    # Merge with fixed params (fixed keys are left as-is; they should already be globally unique)
    combined_parameters = {**trial_params, **config["fixed_parameters"]}
    return combined_parameters



def extract_optuna_trial_params(config: dict):
    assert "fixed_parameters" in config.keys()
    assert "optuna_hp_ranges" in config.keys()
    
    if "trial_params" not in config.keys():
        return config["fixed_parameters"]
    else: 
         return {**config["trial_params"], **config["fixed_parameters"]}        

    

def is_pure_optuna_params_dict(dct): 
    if "fixed_parameters" in dct.keys() and "optuna_hp_ranges" in dct.keys() and len(dct.keys()) == 2: 
        return True 
    elif "fixed_parameters" in dct.keys() and "optuna_hp_ranges" in dct.keys() and  "optuna_hp_options" in dct.keys()  and len(dct.keys()) == 3: 
        True
    else: 
        return False
    
def is_optuna_results_params_dict(dct): 
    if "fixed_parameters" in dct.keys() and "trial_params" in dct.keys(): 
        return True 
    else: 
        return False    

def filter_out_non_dicts(dct): 
    new_dct = {}
    other = {}
    for key, value in dct.items(): 
        if isinstance(value, dict): 
            new_dct[key] = value
        else: 
            other[key] = value
    return new_dct, other


def _stable_choices(choices):
    """
    Make choices deterministic and hashable across trials.
    Keeps original order if it's already deterministic; otherwise returns a tuple.
    """
    # turn into tuple (Optuna is fine with list/tuple)
    try:
        return tuple(choices)
    except TypeError:
        # choices contains unhashables; trust caller to keep it stable
        return tuple(choices)
    


def filter_out_optima_params(config, seperation_classes=["transform_params", "model_params", "optimizer_params", "scheduler_params"]):
    dct_new = {}
    dct_new_not_matching = {}
    
    dcts, other = filter_out_non_dicts(config)
    for k, v in dcts.items():
        if k in seperation_classes:
            dct_new[k] = v
        else: 
            dct_new_not_matching[k] = v

    return dct_new, {**dct_new_not_matching, **other}


def move_matching_params(optuna_params_dict, trial_params, drop_optuna_hp_ranges=False):
    if "trial_params" not in optuna_params_dict.keys():
        optuna_params_dict["trial_params"] = {}
    # parameter with value from trial.params
    names = [optuna_range["params"]["name"]
             for optuna_range in optuna_params_dict["optuna_hp_ranges"]]
    for k in list(trial_params.keys()):
        if k in names:
            optuna_params_dict["trial_params"][k] = trial_params.pop(k)
            if drop_optuna_hp_ranges: 
                optuna_params_dict["optuna_hp_ranges"] = [optuna_range for optuna_range in optuna_params_dict["optuna_hp_ranges"] if optuna_range["params"]["name"] != k]


def recursive_move_matching_params(config_part, trial_params_, drop_optuna_hp_ranges=False):
    for k, v in config_part.items():
        if isinstance(v, dict):
            if is_pure_optuna_params_dict(v):
                move_matching_params(optuna_params_dict=v, trial_params=trial_params_, drop_optuna_hp_ranges=drop_optuna_hp_ranges)
            else:  # might have one at a sublevel
                recursive_move_matching_params(v, trial_params_, drop_optuna_hp_ranges)
        else:
            pass  # dont change the non_dicts


def seperate_optuna_parameters(config: dict, trial_params: dict, seperation_classes=["transform_params", "model_params", "optimizer_params", "scheduler_params"], 
                               drop_optuna_hp_ranges=False):
    
    config_, other_config_params = filter_out_optima_params(deepcopy(config), seperation_classes)
    trial_params_ = trial_params.copy()

    recursive_move_matching_params(config_, trial_params_, drop_optuna_hp_ranges=drop_optuna_hp_ranges)
    assert len(trial_params_) == 0, f"Not all trial paramters could be matched ... {trial_params_}"
    return config_, other_config_params

def recursive_change_optuna_params_to_normal_params(config_part):
    for k, v in config_part.items():
        if isinstance(v, dict):
            if is_optuna_results_params_dict(v):
                config_part[k] = {**v["fixed_parameters"], **v["trial_params"]}
                
            else:  # might have one at a sublevel
                recursive_change_optuna_params_to_normal_params(v)
        else:
            pass  # dont change the non_dicts



def recursive_suggest_trial_parameters(trial, config_part, treat_dot_params_special=True, _prefix=""):
    """
    Walk a config tree. When a leaf is a pure optuna block, generate its trial params.
    Parameter names get prefixed with the hierarchical path to avoid collisions.
    """
    for k, v in list(config_part.items()):
        if isinstance(v, dict):
            if is_pure_optuna_params_dict(v):
                # Build hierarchical prefix like "model.encoder" etc.
                optuna_options = v.get("optuna_options", {})
                #use_prefix = optuna_options.get("use_prefix", True)
                # use_prefix:
                block_prefix = f"{_prefix}.{k}" if _prefix else k
                params_dict = generate_optuna_params(trial, v, prefix=block_prefix)
                #else: 
                #    params_dict = generate_optuna_params(trial, v, prefix=None)

                if treat_dot_params_special:
                    params_dict = update_dot_dicts_with_sub_dicts(params_dict)

                config_part[k] = params_dict
            else:
                new_prefix = f"{_prefix}.{k}" if _prefix else k
                recursive_suggest_trial_parameters(
                    trial, v, treat_dot_params_special=treat_dot_params_special, _prefix=new_prefix
                )



# def recursive_suggest_trial_parameters(trial, config_part, treat_dot_params_special=True):
#     for k, v in config_part.items():
#         if isinstance(v, dict):
#             if is_pure_optuna_params_dict(v):
#                 config_part[k] = generate_optuna_params(trial, v)  # leaf (convert)
#                 if treat_dot_params_special: 
#                     config_part[k] = update_dot_dicts_with_sub_dicts(config_part[k])
#             else:  # might have one at a sublevel
#                 recursive_suggest_trial_parameters(trial, v, treat_dot_params_special=treat_dot_params_special)
#         else:
#             pass  # dont change the non_dicts
    
    
    


def change_optuna_params_to_normal_params(config_with_optuna_parameters: dict, seperation_classes=["transform_params", "model_params", "optimizer_params", "scheduler_params"]):
    config_, other_config_params = filter_out_optima_params(deepcopy(config_with_optuna_parameters), seperation_classes)
    recursive_change_optuna_params_to_normal_params(config_)
    return config_, other_config_params



def get_sub_dicts_from_keys_with_dots_pure_foo(model_params):
    """
    Get keys with one dot (.). Ore than one level is not supported
    """
    keys_to_drop = []
    dicts_to_create = set()
    for key in model_params.keys():
        if "." in key:
            s = key.split(".")
            assert(len(s)) == 2
            dicts_to_create.update({s[0]})
            keys_to_drop.append(key)
    dicts_to_create = {k: {} for k in dicts_to_create}
    for key in model_params.keys():
        if "." in key:
            k, k2 = key.split(".")
            dicts_to_create[k] = {**dicts_to_create[k], **{k2: model_params[key]}}
            
    return keys_to_drop, dicts_to_create


def update_dot_dicts_with_sub_dicts(model_params):
    """
    Deal with paramters like: 'REG_dropout_op_kwargs.p': 0.3
    -> Create dict "REG_dropout_op_kwargs" = {"p": 0.3}  and replace paramter in `model_params`
    """
    keys_to_drop, dicts_to_create = get_sub_dicts_from_keys_with_dots_pure_foo(model_params)
    model_params_new = {k:v for k, v in model_params.items() if k not in keys_to_drop}
    model_params_new = {**model_params_new, **dicts_to_create}
    
    return model_params_new


