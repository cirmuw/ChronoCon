import torch
import mlflow
import torch
import torch.nn as nn

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several,
    check_duplicates_in_dataloader
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    evaluate_and_log_testset_results_AE_v3,
    train_loop_AE_v3
)
from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v4 import (
    evaluate_and_log_testset_results_AE_v4,
    train_loop_AE_v4
)

from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn
import torchvision.transforms.v2 as v2
from ra_utils.training.scores_SHS.model_builders import build_models_AE_v1_and2
import ra_utils.utils.utils_torch
from ra_utils.utils.verbosity_enums import *
import ra_utils.utils.utils

import ra_utils.utils.utils_torch

# ------------------------------------------------------------------
# utility helpers ---------------------------------------------------
def _state_dict_from_uri(uri: str):
    """Load a checkpoint's state-dict from either a file path or MLflow URI."""
    if uri.endswith((".pt", ".pth", ".bin")):          # plain torch file
        return torch.load(uri, map_location="cpu", weights_only=True)
    # otherwise we assume an MLflow model URI, e.g. "runs:/<run_id>/best_model_AE"
    return mlflow.pytorch.load_model(uri).state_dict()

def _load_submodule(submodule, src_sd, prefix, strict=False, msg="", verbose=0):
    """Copy every tensor whose key starts with `prefix` into `submodule`."""
    filtered = {k[len(prefix):]: v for k, v in src_sd.items() if k.startswith(prefix)}
    missing, unexpected = submodule.load_state_dict(filtered, strict=strict)
    if verbose > 1: 
        print(f"{msg}  | loaded={len(filtered):4d}  missing={len(missing):3d}  unexpected={len(unexpected):3d}")
    if verbose > 2:
        if len(filtered) > 0:
            print(f"Loaded keys: {list(filtered.keys())[:5]}...")
        if len(missing) > 0:
            print(f"Missing keys: {missing[:5]}...")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected[:5]}...")


def maybe_partially_init_model_from_state_dict(config: dict, 
                                                model_AE: nn.Module, 
                                                model_c: nn.Module, 
                                                verbose=2):


    # ------------------------------------------------------------------
    # 1) DECODER-ONLY ----------------------------------------------------
    if config.get("model_initialization", {}).get("load_decoder_only", False):
        uri = config["model_initialization"]["pth_src_AE"]
        if verbose: 
            print(f"→ Loading *reconstruction decoder* from {uri}")
        ckpt_sd = _state_dict_from_uri(uri)
        # path inside ckpt: "auto_encoder.decoders.recon.<weight_name>"
        target = model_AE.auto_encoder.decoders["recon"]
        _load_submodule(target, ckpt_sd,
                        prefix="auto_encoder.decoders.recon.",
                        strict=False,
                        msg="decoder", 
                        verbose=verbose)


    # ------------------------------------------------------------------
    # 2) FULL AUTO-ENCODER ----------------------------------------------
    if config.get("model_initialization", {}).get("load_full_AE", False):
        uri = config["model_initialization"]["pth_src_AE"]
        if verbose: 
            print(f"→ Loading FULL auto-encoder from {uri}")
        ckpt_sd = _state_dict_from_uri(uri)

        target = model_AE.auto_encoder                     
        _load_submodule(target, ckpt_sd,
                        prefix="auto_encoder.",
                        strict=False,                     
                        msg="auto_encoder", 
                        verbose=verbose)


    # ------------------------------------------------------------------
    # 3) CLASSIFIER ------------------------------------------------------
    if config.get("load_classifier", {}).get("load_full_AE", False):
        uri = config["model_initialization"]["pth_src_classifier"]
        if verbose: 
            print(f"→ Loading classifier from {uri}")
        ckpt_sd = _state_dict_from_uri(uri)

        _load_submodule(model_c, ckpt_sd,
                        prefix="",                         # whole ckpt *is* the classifier
                        strict=False,
                        msg="classifier", 
                        verbose=verbose)
    return None




def run_training(config: dict,  mlflow_logging=True, verbose=VerboseLevel.CHATTY, 
                 config_name=None,
                 append_BEST_VAL_as_last=False): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose >= VerboseLevel.CHATTY:
        print("Running on ", device)

    #-----------------------------------------------------------------------
    # Load tables with paths and scores (+ split)
    # data_tables = load_img_SHS_patch_data(config["data"])
    data_tables = process_several_score_groups(config["data"])

    # Check that elements exist
    for k,v  in data_tables.items():
        assert len(v["df_include"]) >0, f"score group {k} has no items. Check the image paths or score groups!"
    

    # Make dataset and dataloaders
    # data = dataset_and_loader(data_tables, config)
    data = dataset_and_loader_several(data_tables, config)
    
    if config.get("CheckDL4Duplicates", False): 
        # check_duplicates_in_dataloader(data, ds_key="train_loader")
        print("Checking for duplicates")
        check_duplicates_in_dataloader(data, ds_key="val_loader")
        check_duplicates_in_dataloader(data, ds_key="test_loader")
        print("Done - Checking for duplicates\n ")

    # Load/ make model
    attention_paths_dct = config["data"].get("network_score_groups")
    if attention_paths_dct is None:
        print("'network_score_groups' not found in config file. Using default attention paths: 'score_groups'")
        attention_paths_dct = config["data"]["score_groups"]

    classifier_head_infos = config["data"]["classifier_head_infos"]
    if config["data"].get("how_to_deal_with_surgery")  == "keep: map over limit to limit plus one":
        print("Adding one to the output dimension of the classifier heads! (surgery class)")
        for k in classifier_head_infos.keys():
            v = classifier_head_infos[k]["out_dim"]
            classifier_head_infos[k]["out_dim"] = v + 1

    # If pure regression -> output-dim has to be 1
    classifier_name = config["model"].get("classifier", {}).get("name", "LogReg")
    print(f"{classifier_name = }")
    if classifier_name == "Reg":
        for k in classifier_head_infos.keys():
            classifier_head_infos[k]["out_dim"] = 1
        config["task_type_y"] = "regression"
    elif classifier_name == "LogReg":
        config["task_type_y"] = "classification"

    elif classifier_name == "MixLogAndReg":  # New default!
        raise NotImplementedError("MixLogAndReg is not implemented yet!")
        config["task_type_y"] = "classification_regression_mix"
        for k in classifier_head_infos.keys():
            classifier_head_infos[k]["out_dim"] += 1 # 0 is Regression output; 1...out_dim+1 = class logits                


    model_name = config["model_name"]
    model_AE, model_c = build_models_AE_v1_and2(
                                        model_name, config, 
                                        classifier_head_infos = classifier_head_infos, 
                                        attention_paths_dct = attention_paths_dct
                                        )
    # Maybe also make model_c_reg
    # TODO

    maybe_partially_init_model_from_state_dict(config, model_AE, model_c, verbose=3)


    model_AE.to(device)
    model_c.to(device)

    # define loss function
    loss_fn_y = get_score_loss_function(config["loss"]["score"])
    # loss_fn_y_reg = get_score_loss_function(config["loss"].get("score_reg", {}))
    loss_fn_x = nn.MSELoss()
    loss_fn_z = nn.L1Loss()
    loss_fn_z_triplet_classes = get_triplet_loss_fn(config["loss"].get("triplet_scores", {}))
    
    



    # ---- joint optimizer with separate lrs --------------------------------
    # opt_cfg = config["optimizer_params"].copy()
    # lr_ae  = opt_cfg["learning_rates"]["encoder__OR__decoder"]
    # lr_clf = opt_cfg["learning_rates"]["classifier"]
    # param_groups = [
    #     {"params": model_AE.parameters(), "lr": lr_ae},
    #     {"params": model_c.parameters(),  "lr": lr_clf},
    # ]
    # print(opt_cfg["other_optimizer_kwargs"])
    # optimizer = torch.optim.AdamW(param_groups, **opt_cfg["other_optimizer_kwargs"])
    # # scheduler
    # scheduler = None
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', verbose=True)


    optimizer_params = config["optimizer_params"]
    scheduler_params = config.get("scheduler_params", {})

    optimizer_name = config.get('optimizer_name', "torch.optim.AdamW")
    optimizer_class = ra_utils.utils.utils.pydoc_locate_targets([optimizer_name], chill=False)[0]  
    
    scheduler_name = config.get('scheduler_name', None)
    scheduler_class = ra_utils.utils.utils.pydoc_locate_targets([scheduler_name], chill=False)[0] if scheduler_name else None

    optimizer, scheduler = ra_utils.utils.utils_torch.plan_optimization_v3(
        [model_AE, model_c], # maybe add loss functions if these are trainable
        optimizer_class=optimizer_class, optimizer_params=optimizer_params,
        scheduler_class=scheduler_class, scheduler_params=scheduler_params,
        verbose = (verbose == VerboseLevel.CHATTY)
    )


    
    # #  This works, but not exactly what I want!
    # param_groups = [
    #     {"params": model_AE.parameters()},
    #     {"params": model_c.parameters()},
    # ]
    # optimizer = torch.optim.AdamW(param_groups, lr=1e-4)
    # scheduler = None




    # AE transform 
    AE_transform_name = config.get("AE_transform", {}).get("name")
    if AE_transform_name == None: 
        transform_AE = lambda x: x
        print("NO AE transform!!!")
    elif AE_transform_name == "GaussianNoiseWithClip":    
        sigma = config.get("AE_transform", {}).get("GaussianNoise_sigma", 0.05)
        transform_AE = v2.GaussianNoise(mean=0, sigma = sigma, clip=True)
    else: 
        raise NotImplementedError(f"{AE_transform_name = }")

    train_dataloaders = {k: data[k]["train_loader"] for k in data.keys()}
    val_loaders = {k: data[k]["val_loader"] for k in data.keys()}
    test_loaders = {k: data[k]["test_loader"] for k in data.keys()}
    epochs = config["training"]["epochs"]

    classes = None # get_classes(config) # TO be removed

    print("Start training for: ", config_name)
    model_AE, model_c, metrics_Tr, metrics_Val = train_loop_AE_v3(
        model_AE=model_AE,
        model_classifier=model_c,
        train_dataloaders=train_dataloaders,
        val_loaders=val_loaders,
        loss_fn_x=loss_fn_x,
        loss_fn_y=loss_fn_y,
        loss_fn_z=loss_fn_z,
        loss_fn_z_triplet_classes = loss_fn_z_triplet_classes,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        patience=config["training"].get("early_stopping_tol", 100),
        #
        # Loss function weights
        lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0),   
        lambda_y=config.get('loss_weights', {}).get('lambda_y', 0.0),      
        lambda_z=config.get('loss_weights', {}).get('lambda_z', 0.0),   
        lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0),
        lambda_z_triplet_time = config.get('loss_weights', {}).get('lambda_z_triplet_time', 0.0),
        lambda_z_score_consistency_regularizer = config.get('loss_weights', {}).get('lambda_z_score_consistency_regularizer', 0.0),
        lambda_y_delta = config.get('loss_weights', {}).get('lambda_y_delta', 0.0),   
        lambda_y_reg_extra = config.get('loss_weights', {}).get('lambda_y_reg_extra', 0.0),   
        #
        transform=transform_AE,
        classes=classes,
        log_model_full=config.get("SAVE_MODEL_FULL", False),
        log_model_state_dct = config.get("SAVE_MODEL_state_dct", False),
        verbose=3,
        ES_metric_key=config["training"].get("early_stopping_metric_key", "L"), 
        append_BEST_VAL_as_last=append_BEST_VAL_as_last,
        task_type_y = config.get("task_type_y", "classification")
    )



    artifact_uri = mlflow.get_artifact_uri()
    print("ARTIFACTS URI = ", artifact_uri)
    if config_name != None: 
        print("Done with training for: ", config_name)

    evaluate_on_testset = config.get("evaluate_on_testset", False)
    if evaluate_on_testset:
        print("Evaluating on test set")
        evaluate_and_log_testset_results_AE_v3( # TODO maybe add triplet loss
            model_AE=model_AE,
            model_classifier=model_c,
            dataloaders=test_loaders,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            loss_fn_z_triplet_classes=loss_fn_z_triplet_classes,
            lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0), 
            lambda_y=config.get('loss_weights', {}).get('lambda_y', 0.0),
            lambda_z=config.get('loss_weights', {}).get('lambda_z', 0.0),
            lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0),
            lambda_z_triplet_time = config.get('loss_weights', {}).get('lambda_z_triplet_time', 0.0),
            lambda_z_score_consistency_regularizer = config.get('loss_weights', {}).get('lambda_z_score_consistency_regularizer', 0.0),
            # TODO add lambda_z_score_consistency_regularizer
            lambda_y_delta = config.get('loss_weights', {}).get('lambda_y_delta', 0.0),   
            lambda_y_reg_extra = config.get('loss_weights', {}).get('lambda_y_reg_extra', 0.0),   
            device=device,
            classes=classes,
            transform=transform_AE,
            prefix="test_",
            task_type_y = config.get("task_type_y", "classification")
        )

    if config.get("eval_on_val_set_once_more", True):
        print("Evaluating on val set once more")
        evaluate_and_log_testset_results_AE_v3(
            model_AE=model_AE,
            model_classifier=model_c,
            dataloaders=val_loaders,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0), 
            lambda_y=config.get('loss_weights', {}).get('lambda_y', 0.0),
            lambda_z=config.get('loss_weights', {}).get('lambda_z', 0.0),
            lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0),            
            lambda_y_delta = config.get('loss_weights', {}).get('lambda_y_delta', 0.0),   
            lambda_y_reg_extra = config.get('loss_weights', {}).get('lambda_y_reg_extra', 0.0),               
            loss_fn_z_triplet_classes=loss_fn_z_triplet_classes,
            device=device,
            classes=classes,
            transform=transform_AE,
            prefix="valFinal_",
            skip_metrics_logging=False, # These are already logged in the training loop
            task_type_y = config.get("task_type_y", "classification")
        )

    return {"metrics Val": metrics_Val,
            "metrics Tr": metrics_Tr
            }
    #-----------------------------------------------------------------------

########################################################################
############################### V2  ####################################
########################################################################

import ra_utils.loss.loss_fn_dict

def run_training_v2(config: dict,  mlflow_logging=True, verbose=VerboseLevel.CHATTY, 
                 config_name=None,
                 append_BEST_VAL_as_last=False): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose >= VerboseLevel.CHATTY:
        print("Running on ", device)

    #-----------------------------------------------------------------------
    # Load tables with paths and scores (+ split)
    # data_tables = load_img_SHS_patch_data(config["data"])
    data_tables = process_several_score_groups(config["data"])

    # Check that elements exist
    for k,v  in data_tables.items():
        assert len(v["df_include"]) >0, f"score group {k} has no items. Check the image paths or score groups!"
    

    # Make dataset and dataloaders
    # data = dataset_and_loader(data_tables, config)
    data = dataset_and_loader_several(data_tables, config)
    
    if config.get("CheckDL4Duplicates", False): 
        # check_duplicates_in_dataloader(data, ds_key="train_loader")
        print("Checking for duplicates")
        check_duplicates_in_dataloader(data, ds_key="val_loader")
        check_duplicates_in_dataloader(data, ds_key="test_loader")
        print("Done - Checking for duplicates\n ")

    # Load/ make model
    attention_paths_dct = config["data"].get("network_score_groups")
    if attention_paths_dct is None:
        print("'network_score_groups' not found in config file. Using default attention paths: 'score_groups'")
        attention_paths_dct = config["data"]["score_groups"]

    classifier_head_infos = config["data"]["classifier_head_infos"]
    if config["data"].get("how_to_deal_with_surgery")  == "keep: map over limit to limit plus one":
        print("Adding one to the output dimension of the classifier heads! (surgery class)")
        for k in classifier_head_infos.keys():
            v = classifier_head_infos[k]["out_dim"]
            classifier_head_infos[k]["out_dim"] = v + 1

    # If pure regression -> output-dim has to be 1
    classifier_name = config["model"].get("classifier", {}).get("name", "LogReg")
    print(f"{classifier_name = }")
    if classifier_name == "Reg":
        for k in classifier_head_infos.keys():
            classifier_head_infos[k]["out_dim"] = 1
        config["task_type_y"] = "regression"
    elif classifier_name == "LogReg":
        config["task_type_y"] = "classification"

    elif classifier_name == "MixLogAndReg":  # New default!
        raise NotImplementedError("MixLogAndReg is not implemented yet!")
        config["task_type_y"] = "classification_regression_mix"
        for k in classifier_head_infos.keys():
            classifier_head_infos[k]["out_dim"] += 1 # 0 is Regression output; 1...out_dim+1 = class logits                


    model_name = config["model_name"]
    model_AE, model_c = build_models_AE_v1_and2(
                                        model_name, config, 
                                        classifier_head_infos = classifier_head_infos, 
                                        attention_paths_dct = attention_paths_dct
                                        )

    maybe_partially_init_model_from_state_dict(config, model_AE, model_c, verbose=3)
    model_AE.to(device)
    model_c.to(device)

    loss_fn_dict = ra_utils.loss.loss_fn_dict.get_loss_fn_dict(config, device=device)



    optimizer_params = config["optimizer_params"]
    scheduler_params = config.get("scheduler_params", {})

    optimizer_name = config.get('optimizer_name', "torch.optim.AdamW")
    optimizer_class = ra_utils.utils.utils.pydoc_locate_targets([optimizer_name], chill=False)[0]  
    
    scheduler_name = config.get('scheduler_name', None)
    scheduler_class = ra_utils.utils.utils.pydoc_locate_targets([scheduler_name], chill=False)[0] if scheduler_name else None

    optimizer, scheduler = ra_utils.utils.utils_torch.plan_optimization_v3(
        [model_AE, model_c], # maybe add loss functions if these are trainable
        optimizer_class=optimizer_class, optimizer_params=optimizer_params,
        scheduler_class=scheduler_class, scheduler_params=scheduler_params,
        verbose = (verbose == VerboseLevel.CHATTY)
    )



    # AE transform 
    AE_transform_name = config.get("AE_transform", {}).get("name")
    if AE_transform_name == None: 
        transform_AE = lambda x: x
        print("NO AE transform!!!")
    elif AE_transform_name == "GaussianNoiseWithClip":    
        sigma = config.get("AE_transform", {}).get("GaussianNoise_sigma", 0.05)
        transform_AE = v2.GaussianNoise(mean=0, sigma = sigma, clip=True)
    else: 
        raise NotImplementedError(f"{AE_transform_name = }")

    train_dataloaders = {k: data[k]["train_loader"] for k in data.keys()}
    val_loaders = {k: data[k]["val_loader"] for k in data.keys()}
    test_loaders = {k: data[k]["test_loader"] for k in data.keys()}
    epochs = config["training"]["epochs"]



    print("Start training for: ", config_name)
    model_AE, model_c, metrics_Tr, metrics_Val = train_loop_AE_v4(
        model_AE=model_AE,
        model_classifier=model_c,
        train_dataloaders=train_dataloaders,
        val_loaders=val_loaders,
        loss_fn_dict=loss_fn_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        patience=config["training"].get("early_stopping_tol", 100),
        transform=transform_AE,
        log_model_full=config.get("SAVE_MODEL_FULL", False),
        log_model_state_dct = config.get("SAVE_MODEL_state_dct", False),
        verbose=3,
        ES_metric_key=config["training"].get("early_stopping_metric_key", "L"), 
        append_BEST_VAL_as_last=append_BEST_VAL_as_last,
        task_type_y = config.get("task_type_y", "classification")
    )



    artifact_uri = mlflow.get_artifact_uri()
    print("ARTIFACTS URI = ", artifact_uri)
    if config_name != None: 
        print("Done with training for: ", config_name)

    evaluate_on_testset = config.get("evaluate_on_testset", False)
    if evaluate_on_testset:
        print("Evaluating on test set")
        evaluate_and_log_testset_results_AE_v4( # TODO maybe add triplet loss
            model_AE=model_AE,
            model_classifier=model_c,
            dataloaders=test_loaders,
            loss_fn_dict=loss_fn_dict,
            device=device,
            transform=transform_AE,
            prefix="test_",
            task_type_y = config.get("task_type_y", "classification")
        )

    if config.get("eval_on_val_set_once_more", True):
        print("Evaluating on val set once more")
        evaluate_and_log_testset_results_AE_v4(
            model_AE=model_AE,
            model_classifier=model_c,
            dataloaders=val_loaders,
            loss_fn_dict=loss_fn_dict,
            device=device,
            transform=transform_AE,
            prefix="valFinal_",
            skip_metrics_logging=False, # These are already logged in the training loop
            task_type_y = config.get("task_type_y", "classification")
        )

    return {"metrics Val": metrics_Val,
            "metrics Tr": metrics_Tr
            }
    #-----------------------------------------------------------------------




def cuda_oom_cleanup(func):
    """
    Decorator to catch CUDA out of memory errors, cleanup CUDA cache and re-raise the error.
    
    Args:
        func: The function to decorate
        
    Returns:
        The wrapped function that handles CUDA OOM errors
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise e
    
    return wrapper


run_training_with_cleanup = cuda_oom_cleanup(run_training)

