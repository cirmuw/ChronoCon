from typing import List, Tuple, Callable, Optional
from typing import Tuple, Dict            
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer, SGD
import torch 
import torch.nn as nn 

import ra_utils.training.lr_scheduler.polylr
from ra_utils.training.lr_scheduler.polylr import (
    PolyLRScheduler,
    PolyLRSchedulerMultiLR
)




class NamesFilter():
    def __init__(self, names_filter: Callable | str = lambda name: True) -> None:
        if callable(names_filter):
            self.filter_str_rep = "user provided function"
            filter_fn = names_filter
        
        elif isinstance(names_filter, str):
            # e.g. "encoder"  OR  "encoder|decoder"
            parts = names_filter.split('__OR__')
            
            if len(parts) > 1:
                # If multiple parts are detected, match any of them.
                self.filter_str_rep = f"starts with any of {parts}"
                filter_fn = lambda name: any(name.startswith(part) for part in parts)
            else:
                # If there's no '|', then match the single prefix.
                self.filter_str_rep = f"starts with '{names_filter}'"
                filter_fn = lambda name: name.startswith(names_filter)
        
        else:
            raise ValueError(f"Unexpected type given: {type(names_filter)} — only str or function are allowed.")

        self.filter_fn = filter_fn
        # self.matched = None
        # self.not_matched = None
        self.matched = []
        self.not_matched = []

    def __repr__(self) -> str:
        return f"NamesFilter: {self.filter_str_rep}"

    def __call__(self, xl: List[Tuple[str, torch.nn.parameter.Parameter]]) -> None:
        # self.matched = []
        # self.not_matched = []

        for x_name, x_param in xl:
            if self.filter_fn(x_name):
                self.matched.append((x_name, x_param))
            else:
                self.not_matched.append((x_name, x_param))

        # For cascade usage, return the elements for which the filter did *not* match
        return self.not_matched

    
def devide_model_paramters(model, filters: List[NamesFilter], verbose = False) -> List[NamesFilter]:
    """ Devide model parameters into groups.
    
    This was built for a layer-specific learning rate, ... 
    
    Examples:
    ---------
    >>> filters = [
        NamesFilter(lambda name: name.startswith("conv")),  # get all layers starting with 'conv'
        NamesFilter(""fc",    # -> get all layers not starting with ' conv' but starting with 'fc'
        NamesFilter(),                                      # -> get all other layes
    ]

    >>> filters = devide_model_paramters(model, filters, verbose=False)    
    >>> opt_params = [{"params": [fi[1] for fi in filters[0].matched], "lr":1.0E-5},   # doctest: +SKIP
                      {"params": [fi[1] for fi in filters[1].matched], "lr":1.0E-4},
                      {"params": [fi[1] for fi in filters[2].matched]}]   # default

    >>> optimizer = optim.SGD(opt_params, lr=0.001, momentum=0.9)   # doctest: +SKIP
    """
    current_params = model.named_parameters()    
    for filter in filters: 
        current_params = filter(current_params)
        
    missed_names = [c[0] for c in current_params]
    assert len(current_params) == 0, f"Some params were missed: {missed_names}"
    
    # updated filters: 
    if verbose:
        for i, filter in enumerate(filters):
            print(f"Group: {i}:", [x[0] for x in filter.matched])
            print(f"    Filter: {filter}")            
    return filters 
            

def collect_unique_named_parameters(models):
    """Flatten and deduplicate named params from one or more models by object identity."""
    seen = set()
    named = []
    for m in models:
        if m is None:
            continue
        for n, p in m.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            named.append((n, p))
    return named


def divide_params_by_filters_once(named_params, filters, *, verbose=False, catch_rest=False):
    # reset buffers for a clean run
    for f in filters:
        f.matched = []
        f.not_matched = []

    remaining = list(named_params)

    # sequentially carve out matches
    for f in filters:
        remaining = f(remaining)

    if verbose:
        for i, f in enumerate(filters):
            print(f"[group {i}] matched {len(f.matched)} params")

    # sanity check for accidental overlaps across groups
    seen_ids = set()
    for f in filters:
        for _, p in f.matched:
            pid = id(p)
            if pid in seen_ids:
                raise RuntimeError("A parameter matched multiple filters — fix overlapping filters.")
            seen_ids.add(pid)

    # don't assert on leftovers here; return them so caller can decide
    return filters, remaining 


def plan_optimization(model, 
                      learning_rates: Dict[str, float],
                      max_steps=10,
                      weight_decay: float =3.0E-5,
                      default_lr: float = 1.0E-3,
                      momentum: float =0.99
                      ) -> Tuple[Optimizer, LRScheduler]: 
    # get filters from learning rate dictionary
    filters = []
    for layer_name in learning_rates.keys():
        #print(layer_name)
        filters.append(NamesFilter(layer_name))
    filters.append(NamesFilter(lambda x: True)) # default (catch all the rest)

    filters = devide_model_paramters(model, filters=filters, verbose=True)

    opt_params = []
    for i, initial_lr in enumerate(learning_rates.values()):
        if initial_lr > 1.0E-10:
            opt_params.append({"params": [fi[1] for fi in filters[i].matched], "lr": initial_lr})
        else:
            print(f"For group {i} the initial lr is {initial_lr} so it is not passed to optimizer")
            print(f"   named parameters: ", [fi[0] for fi in filters[i].matched])

    optimizer = SGD(opt_params, lr=default_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = PolyLRSchedulerMultiLR(optimizer, max_steps=max_steps)

    return optimizer, scheduler


optimizer_default_params = {
    "learning_rates": {
        "encoder.stages.0": 0.0,
        "encoder.stages.1": 0.0,
        "encoder.stages.2": 0.0,
        "encoder.stages.3": 1.0E-5,
        "encoder.stages.4": 1.0E-5,
        "encoder.stages.5": 1.0E-5,
        "mlp": 1.0E-4,
        "projector": 1.0E-4
    },
    "other_optimizer_kwargs": {
        "weight_decay": 3.0E-5,
        "lr": 1.0E-3,
        "momentum": 0.99,
        "nesterov": True
    }
}

def plan_optimization_v2(model, 
                         optimizer_class = SGD,
                         optimizer_params = optimizer_default_params,
                         scheduler_class = None, #PolyLRSchedulerMultiLR,
                         scheduler_params = {"max_steps": 10},
                         verbose=False, 
                         catch_rest = False
                        ) -> Tuple[Optimizer, LRScheduler]: 
    
    learning_rates = optimizer_params["learning_rates"]
    other_optimizer_kwargs = optimizer_params["other_optimizer_kwargs"]
    
    # get filters from learning rate dictionary
    filters = []
    for layer_name in learning_rates.keys():
        #print(layer_name)
        filters.append(NamesFilter(layer_name))
    if catch_rest: 
        filters.append(NamesFilter(lambda x: True)) # default (catch all the rest)
    filters = devide_model_paramters(model, filters=filters, verbose=verbose)


    opt_params_ = []
    for i, initial_lr in enumerate(learning_rates.values()):
        if initial_lr > 1.0E-10:
            opt_params_.append({"params": [fi[1] for fi in filters[i].matched], "lr": initial_lr})
        else:
            if verbose: 
                print(f"For group {i} the initial lr is {initial_lr} so it is not passed to optimizer")
                print(f"   named parameters: ", [fi[0] for fi in filters[i].matched])

    optimizer = optimizer_class(opt_params_, **other_optimizer_kwargs)
    
    if scheduler_class != None:
        scheduler = scheduler_class(optimizer, **scheduler_params)
    else: 
        scheduler = None

    return optimizer, scheduler


def plan_optimization_v3(models,  # Now takes a list of models  
                         optimizer_class = SGD,
                         optimizer_params = optimizer_default_params,
                         scheduler_class = None, # e.g., PolyLRSchedulerMultiLR
                         scheduler_params = {"max_steps": 10},
                         verbose=False, 
                         catch_rest = False, 
                         batch_size_for_lr_rescaling = None,
                         batch_size_for_lr_rescaling_base = 256, 
                        ) -> Tuple[Optimizer, Optional[LRScheduler]]: 
    
    # Extract from optimizer_params
    learning_rates = optimizer_params["learning_rates"]
    if batch_size_for_lr_rescaling != None: 
        factor = batch_size_for_lr_rescaling / batch_size_for_lr_rescaling_base
        print(f"Rescaling learning rates (batch_size dependent)  {factor = }")
        learning_rates = {k: v*factor for k,v in learning_rates.items()}
    
    other_optimizer_kwargs = optimizer_params["other_optimizer_kwargs"]

    # Create filters based on the keys in the learning_rates dictionary
    filters = []
    for layer_name in learning_rates.keys():
        filters.append(NamesFilter(layer_name))
    if catch_rest:
        # A filter that matches everything else not matched by previous filters
        filters.append(NamesFilter(lambda x: True)) 
    
    # Apply the filters to each model. We assume devide_model_paramters can be called sequentially.
    # After processing all models, filters[i].matched will contain tuples (param_name, param)
    # for all parameters across all models that match that filter.
    for i, model in enumerate(models):
        if verbose: 
            print(f"Model: {i}")
        filters = devide_model_paramters(model, filters=filters, verbose=verbose)
        if verbose: 
            print()

    # Now create the parameter groups for the optimizer
    opt_params_ = []
    # learning_rates and filters are aligned by order:
    # filters[i] corresponds to the i-th entry in learning_rates    
    lr_values = list(learning_rates.values())  # To ensure consistent indexing
    for i, initial_lr in enumerate(lr_values):
        # If the learning rate is very small, we might skip
        # adding that group to the optimizer if desired.
        if initial_lr > 1.0E-10:
            group_params = [fi[1] for fi in filters[i].matched]
            if len(group_params) > 0:
                opt_params_.append({"params": group_params, "lr": initial_lr})
            else:
                if verbose:
                    print(f"For group {i}, no parameters matched. Skipping this group.")
        else:
            if verbose:
                print(f"For group {i} the initial lr is {initial_lr}, not passed to optimizer")
                print("   named parameters: ", [fi[0] for fi in filters[i].matched])

    # Initialize optimizer
    optimizer = optimizer_class(opt_params_, **other_optimizer_kwargs)

    # Initialize scheduler if specified
    if scheduler_class is not None:
        scheduler = scheduler_class(optimizer, **scheduler_params)
    else:
        scheduler = None

    return optimizer, scheduler


def X_y_to_device(X, y, device='cpu'):
    if isinstance(X, list):
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    return X, y



def plan_optimization_v4(models,
                         optimizer_class=SGD,
                         optimizer_params=None,
                         scheduler_class=None,
                         scheduler_params={"max_steps": 10},
                         verbose=False,
                         catch_rest=False,
                         batch_size_for_lr_rescaling=None,
                         batch_size_for_lr_rescaling_base=256,
                         return_report=True,   # <-- NEW
                        ):
    if optimizer_params is None:
        raise ValueError("optimizer_params must be provided")

    learning_rates = optimizer_params["learning_rates"]
    if batch_size_for_lr_rescaling is not None:
        factor = batch_size_for_lr_rescaling / batch_size_for_lr_rescaling_base
        learning_rates = {k: v * factor for k, v in learning_rates.items()}
        if verbose:
            print(f"Rescaling learning rates by factor {factor:.4f}")

    other_optimizer_kwargs = optimizer_params["other_optimizer_kwargs"]

    # Build filters in the same order as learning_rates
    layer_keys = list(learning_rates.keys())
    filters = [NamesFilter(k) for k in layer_keys]
    if catch_rest:
        filters.append(NamesFilter(lambda _: True))

    named_params = collect_unique_named_parameters(models)

    # Divide once and get leftovers
    filters, remaining = divide_params_by_filters_once(
        named_params, filters, verbose=verbose, catch_rest=catch_rest
    )

    # Build optimizer groups
    opt_params_ = []
    per_group = []
    for i, key in enumerate(layer_keys):
        lr_val = float(learning_rates[key])
        group_params = [p for _, p in filters[i].matched]
        per_group.append({
            "group_idx": i,
            "key": key,
            "lr": lr_val,
            "num_params": len(group_params),
            "names": [n for n, _ in filters[i].matched],
        })
        if lr_val > 1e-10 and group_params:
            opt_params_.append({"params": group_params, "lr": lr_val})
        elif verbose and not group_params:
            print(f"[group {i}:{key}] no parameters matched.")

    optimizer = optimizer_class(opt_params_, **other_optimizer_kwargs)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class is not None else None

    # Build a report for unmatched params
    unmatched_trainable = [n for n, p in remaining if p.requires_grad]
    unmatched_frozen = [n for n, p in remaining if not p.requires_grad]

    report = {
        "total_params": len(named_params),
        "matched_params": sum(len(g["names"]) for g in per_group),
        "unmatched_trainable_count": len(unmatched_trainable),
        "unmatched_frozen_count": len(unmatched_frozen),
        "unmatched_trainable_names": unmatched_trainable,
        "unmatched_frozen_names": unmatched_frozen,
        "groups": per_group,
    }

    if verbose:
        print(f"[report] total={report['total_params']} matched={report['matched_params']} "
              f"unmatched_trainable={report['unmatched_trainable_count']} "
              f"unmatched_frozen={report['unmatched_frozen_count']}")
        if report["unmatched_trainable_count"]:
            print("  Unmatched TRAINABLE params (sample):", unmatched_trainable[:10])
        if report["unmatched_frozen_count"]:
            print("  Unmatched FROZEN params (sample):   ", unmatched_frozen[:10])

    return (optimizer, scheduler, report) if return_report else (optimizer, scheduler)