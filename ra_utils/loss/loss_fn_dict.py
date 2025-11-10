import torch
import torch
import torch.nn as nn


from ra_utils.networks.loss_function import (
    get_score_loss_function, 
    get_triplet_loss_fn, 
    get_consistency_regularization_loss_fn,
    get_triplet_loss_fn_WST, 
    get_triplet_loss_fn_MDP, 
    get_triplet_loss_RnCids, 
    get_delta_head_loss
)
import ra_utils.loss.online_mining_delta_loss
from ra_utils.networks.architecture import (
    DummyReturnZeroLoss,
    DummyReturnZeroLossMulti, 
    DummyReturnZeroLossAndDict
)

import ra_utils.loss.loss_RnC_with_ranks_and_ids
import ra_utils.loss.delta_head_utils
import ra_utils.networks.delta_heads


def make_loss_function_dict_delta_heads(config, data, delta_head_infos=None, device="cuda"):
    
    
    lambda_ = config.get('loss_weights', {}).get('lambda_y_DeltaHead', 0.0)
    if lambda_ < 1.0e-8: 
        loss_fn_ = DummyReturnZeroLossAndDict(device)
    else: 
        prevalence_by_score_type = ra_utils.loss.delta_head_utils.score_difference_prevalence_by_score_type(data)


        options_prevalence_weighting  = config["loss"].get("DeltaHead", {}).get("prevalence_weighting")
        if options_prevalence_weighting: 
            prevalences_w = ra_utils.loss.delta_head_utils.prevalences_weights(prevalence_by_score_type, **options_prevalence_weighting)
            # Convert all class weight arrays to torch tensors on the specified device, in a new dict. 
            # This avoids any inplace mutation or race conditions.
            prevalences_w = {k: torch.as_tensor(v, device=device) for k, v in prevalences_w.items()}
        else: 
            prevalences_w = None
            
        loss_params = config["loss"].get("DeltaHead", {}).get("params", {})

        loss_fn_ = ra_utils.networks.delta_heads.DeltaHeadsLoss(delta_head_infos=delta_head_infos, class_weights = prevalences_w, **loss_params)
    loss_dct_y_DeltaHead = {
            "function": loss_fn_, 
            "lambda": lambda_, 
            "options": config["loss"].get("delta_head_loss", {}).get("options", {})
        }    
        
    return loss_dct_y_DeltaHead


def get_loss_fn_dict(config, device="cuda"):
    # Naming convention 
    #   input: X  = image 
    #   output: y = score 
    #   latent rep.: z = encoder(X)

    dummy = DummyReturnZeroLoss(device)
    dummy_with_dict = DummyReturnZeroLossAndDict(device)
    dummy2 = DummyReturnZeroLossMulti(device=device, size=2)
    dummy3 = DummyReturnZeroLossMulti(device=device, size=3)    

    # Loss on the score 
    lambda_ = config.get('loss_weights', {}).get('lambda_y', 0.0)
    if lambda_ < 1.0e-8: 
        loss_fn_ = dummy
    else: 
        loss_fn_ = get_score_loss_function(config["loss"]["score"])
    loss_dct_y = {
            "function": loss_fn_, 
            "lambda": lambda_, 
            "options": config["loss"].get("score", {}).get("options", {})
        }
    

    # reconstruction loss 
    lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0)
    loss_fn_x = nn.MSELoss()
    if lambda_x < 1.0e-8: 
            loss_fn_x = dummy
    loss_dct_x = {
            "function": loss_fn_x, 
            "lambda": lambda_x, 
            "options": {} #
        }
    

    # Latent representation loss 
    # Sparseness
    lambda_z=config.get('loss_weights', {}).get('lambda_z', 0.0)   
    loss_fn_z = nn.L1Loss()
    if lambda_z < 1.0e-8: 
            loss_fn_z = dummy
    loss_dct_z = {
            "function": loss_fn_z, 
            "lambda": lambda_z, 
            "options": {}
        }


    # Triplet loss on scores and Instance Id
    lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0)
    if lambda_z_triplet_classes < 1.0e-8: 
            loss_fn_z_triplet_classes = dummy3
    else: 
        loss_fn_z_triplet_classes = get_triplet_loss_fn(config["loss"].get("triplet_scores_classes", {}))
    loss_dct_z_triplet_classes = {
            "function": loss_fn_z_triplet_classes, 
            "lambda": lambda_z_triplet_classes, 
            "options": config["loss"].get("triplet_scores_classes", {}).get("options", {})
        }
    
    # Triplet loss with self transform (WST) on scores
    lambda_z_triplet_WST_score = config.get('loss_weights', {}).get('lambda_z_triplet_WST_scores', 0.0)
    if lambda_z_triplet_WST_score < 1.0e-8: 
        loss_fn_z_triplet_WST_score = dummy3
    else: 
        loss_fn_z_triplet_WST_score = get_triplet_loss_fn_WST(config["loss"].get("triplet_WST_scores", {}))
    loss_dct_z_triplet_WST_score = {
            "function": loss_fn_z_triplet_WST_score, 
            "lambda": lambda_z_triplet_WST_score, 
            "options": config["loss"].get("triplet_WST_scores", {}).get("options", {})
        }
    
    # # Triplet loss with self transform (WST) on time
    lambda_z_triplet_WST_time = config.get('loss_weights', {}).get('lambda_z_triplet_WST_time', 0.0)
    if lambda_z_triplet_WST_time < 1.0e-8: 
            loss_fn_z_triplet_WST_time = dummy3
    else: 
        loss_fn_z_triplet_WST_time = get_triplet_loss_fn_WST(config["loss"].get("triplet_WST_time", {}))
    loss_dct_z_triplet_WST_time = {
            "function": loss_fn_z_triplet_WST_time, 
            "lambda": lambda_z_triplet_WST_time, 
            "options": config["loss"].get("triplet_WST_time", {}).get("options", {})
        }
    

    lambda_z_triplet_MDP_time = config.get('loss_weights', {}).get('lambda_z_triplet_MDP', 0.0)
    if lambda_z_triplet_MDP_time < 1.0e-8: 
        loss_fn_z_triplet_MDP_time = dummy3
    else: 
        loss_fn_z_triplet_MDP_time = get_triplet_loss_fn_MDP(config["loss"].get("triplet_MDP", {}))
    loss_dct_z_triplet_MDP_time = {
            "function": loss_fn_z_triplet_MDP_time, 
            "lambda": lambda_z_triplet_MDP_time, 
            "options": config["loss"].get("triplet_MDP", {}).get("options", {})
        }    


    lambda_ = config.get('loss_weights', {}).get('lambda_z_RnC_time', 0.0)
    if lambda_ < 1.0e-8: 
            loss_fn_ = dummy_with_dict
    else: 
        loss_fn_ = get_triplet_loss_RnCids(config["loss"].get("RnC_time", {}))
    loss_dct_z_RnC_time = {
            "function": loss_fn_, 
            "lambda": lambda_, 
            "options": config["loss"].get("RnC_time", {}).get("options", {})
        }


    lambda_ = config.get('loss_weights', {}).get('lambda_z_RnC_score', 0.0)
    if lambda_ < 1.0e-8: 
        loss_fn_ = dummy_with_dict
    else: 
        loss_fn_ = get_triplet_loss_RnCids(config["loss"].get("RnC_score", {}))
    loss_dct_z_RnC_score = {
            "function": loss_fn_, 
            "lambda": lambda_, 
            "options": config["loss"].get("RnC_score", {}).get("options", {})
        }


    # lambda_ = config.get('loss_weights', {}).get('lambda_y_delta', 0.0)
    # if lambda_ < 1.0e-8: 
    #     loss_fn_ = dummy_with_dict
    # else: 
    #     loss_fn_ = get_triplet_loss_RnCids(config["loss"].get("RnC_score", {}))
    # loss_dct_z_RnC_score = {
    #         "function": loss_fn_, 
    #         "lambda": lambda_, 
    #         "options": config["loss"].get("RnC_score", {}).get("options", {})
    #     }



    # lambda_ = config.get('loss_weights', {}).get('lambda_z_RnC_score_mono', 0.0)
    # if lambda_ < 1.0e-8: 
    #     loss_fn_ = dummy_with_dict
    # else: 
    #     loss_fn_ = get_triplet_loss_RnCids(config["loss"].get("RnC_score_mono", {}))
    # loss_dct_z_RnC_score_mono = {
    #         "function": loss_fn_, 
    #         "lambda": lambda_, 
    #         "options": config["loss"].get("RnC_score_mono", {}).get("options", {})
    #     }



    # Score consistency loss (similar to https://arxiv.org/html/2508.00496v2)
    lambda_z_score_consistency_regularizer = config.get('loss_weights', {}).get('lambda_z_score_consistency_regularizer', 0.0)
    if lambda_z_score_consistency_regularizer < 1.0e-8: 
            loss_fn_z_score_consistency_regularizer = dummy2
    else: 
            loss_fn_z_score_consistency_regularizer = get_consistency_regularization_loss_fn(config["loss"].get("score_consistency_regularizer", {})) 
    loss_dct_z_score_consistency_regularizer = {
            "function": loss_fn_z_score_consistency_regularizer, 
            "lambda": lambda_z_score_consistency_regularizer, 
            "options": config["loss"].get("score_consistency_regularizer", {}).get("options", {})
        }

    # Duplet loss (e.g. Huber loss)
    lambda_y_delta = config.get('loss_weights', {}).get('lambda_y_delta', 0.0)
    if lambda_y_delta < 1.0e-8: 
        loss_fn_y_delta = dummy
    else: 
        loss_fn_y_delta = ra_utils.loss.online_mining_delta_loss.batch_all_score_differences_loss
    loss_dct_y_delta = {
        "function": loss_fn_y_delta,
        "lambda": lambda_y_delta,
        "options": {} # config["loss"].get("XXXX", {}).get("options", {})
    }


    # lambda_ = config.get('loss_weights', {}).get('lambda_y_DeltaHead', 0.0)
    # if lambda_ < 1.0e-8: 
    #     loss_fn_ = dummy_with_dict
    # else: 
    #     loss_fn_ = get_delta_head_loss(config["loss"].get("DeltaHead", {}))
    # loss_dct_y_DeltaHead = {
    #         "function": loss_fn_, 
    #         "lambda": lambda_, 
    #         "options": config["loss"].get("delta_head_loss", {}).get("options", {})
    #     }



    lambda_y_reg_extra = config.get('loss_weights', {}).get('lambda_y_reg_extra', 0.0)
    if lambda_y_reg_extra < 1.0e-8: 
        loss_fn_y_reg_extra = dummy
    else: 
        loss_fn_y_reg_extra = nn.MSELoss()
    loss_dct_y_reg_extra = {
        "function": loss_fn_y_reg_extra,
        "lambda": lambda_y_reg_extra,
        "options": {}# config["loss"].get("XXXX", {}).get("options", {})
    }

    r = {
        "x": loss_dct_x,
        #
        "y": loss_dct_y,
        "y_delta": loss_dct_y_delta,
        "y_reg_extra": loss_dct_y_reg_extra,
        #
        #"y_DeltaHead": loss_dct_y_DeltaHead,
        # 
        "z": loss_dct_z,
        #
        "y_delta": loss_dct_y_delta,
        "y_reg_extra": loss_dct_y_reg_extra,
        #
        "z_triplet_classes": loss_dct_z_triplet_classes,
        "z_triplet_WST_score": loss_dct_z_triplet_WST_score, 
        "z_triplet_WST_time": loss_dct_z_triplet_WST_time,   # Later ... or not 
        #
        "z_score_consistency_regularizer": loss_dct_z_score_consistency_regularizer,
        #
        "z_triplet_MDP_time": loss_dct_z_triplet_MDP_time,
        "z_RnC_score": loss_dct_z_RnC_score,
        #"z_RnC_score_mono": loss_dct_z_RnC_score_mono,
        "z_RnC_time": loss_dct_z_RnC_time
    }

    return r
