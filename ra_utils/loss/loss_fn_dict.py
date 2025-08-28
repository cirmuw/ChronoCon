import torch
import torch
import torch.nn as nn


from ra_utils.networks.loss_function import (
    get_score_loss_function, 
    get_triplet_loss_fn, 
    get_consistency_regularization_loss_fn,
    get_triplet_loss_fn_WST
)
import ra_utils.loss.online_mining_delta_loss
from ra_utils.networks.architecture import (
    DummyReturnZeroLoss,
    DummyReturnZeroLossMulti
)



def get_loss_fn_dict(config, device="cuda"):
    # Naming convention 
    #   input: X  = image 
    #   output: y = score 
    #   latent rep.: z = encoder(X)

    dummy = DummyReturnZeroLoss(device)
    dummy2 = DummyReturnZeroLossMulti(device=device, size=2)
    dummy3 = DummyReturnZeroLossMulti(device=device, size=3)    

    # Loss on the score 
    lambda_y=config.get('loss_weights', {}).get('lambda_y', 0.0)    
    loss_fn_y = get_score_loss_function(config["loss"]["score"])
    loss_dct_y = {
            "function": loss_fn_y, 
            "lambda": lambda_y, 
            "options": None
        }
    

    # reconstruction loss 
    lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0)
    loss_fn_x = nn.MSELoss()
    if lambda_x < 1.0e-8: 
            loss_fn_x = dummy
    loss_dct_x = {
            "function": loss_fn_x, 
            "lambda": lambda_x, 
            "options": None
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
            "options": None
        }


    # Triplet loss on scores and Instance Id
    lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0)
    loss_fn_z_triplet_classes = get_triplet_loss_fn(config["loss"].get("triplet_scores_classes", {}))
    if lambda_z_triplet_classes < 1.0e-8: 
            loss_fn_z_triplet_classes = dummy3
    loss_dct_z_triplet_classes = {
            "function": loss_fn_z_triplet_classes, 
            "lambda": lambda_z_triplet_classes, 
            "options": None
        }
    
    # Triplet loss with self transform (WST) on scores
    lambda_z_triplet_WST_score = config.get('loss_weights', {}).get('lambda_z_triplet_WST_scores', 0.0)
    loss_fn_z_triplet_WST_score = get_triplet_loss_fn_WST(config["loss"].get("triplet_WST_scores", {}))
    if lambda_z_triplet_WST_score < 1.0e-8: 
            loss_fn_z_triplet_WST_score = dummy3
    loss_dct_z_triplet_WST_score = {
            "function": loss_fn_z_triplet_WST_score, 
            "lambda": lambda_z_triplet_WST_score, 
            "options": None
        }
    
    # # Triplet loss with self transform (WST) on time
    # lambda_z_triplet_WST_time = config.get('loss_weights', {}).get('lambda_z_triplet_WST_time', 0.0)
    # loss_fn_z_triplet_WST_time = get_triplet_loss_fn_WST(config["loss"].get("triplet_WST_time", {}))
    # if lambda_z_triplet_WST_time < 1.0e-8: 
    #         loss_fn_z_triplet_WST_time = dummy2
    # loss_dct_z_triplet_WST_time = {
    #         "function": loss_fn_z_triplet_WST_time, 
    #         "lambda": lambda_z_triplet_WST_time, 
    #         "options": None
    #     }


    # Score consistency loss (similar to https://arxiv.org/html/2508.00496v2)
    lambda_z_score_consistency_regularizer = config.get('loss_weights', {}).get('lambda_z_score_consistency_regularizer', 0.0)
    loss_fn_z_score_consistency_regularizer = get_consistency_regularization_loss_fn(config["loss"].get("score_consistency_regularizer", {}))
    if lambda_z_score_consistency_regularizer < 1.0e-8: 
            loss_fn_z_score_consistency_regularizer = dummy
    loss_dct_z_score_consistency_regularizer = {
            "function": loss_fn_z_score_consistency_regularizer, 
            "lambda": lambda_z_score_consistency_regularizer, 
            "options": None
        }

    # Duplet loss (e.g. Huber loss)
    lambda_y_delta = config.get('loss_weights', {}).get('lambda_y_delta', 0.0)
    loss_fn_y_delta = ra_utils.loss.online_mining_delta_loss.batch_all_score_differences_loss
    if lambda_y_delta < 1.0e-8: 
        loss_fn_y_delta = dummy
    loss_dct_y_delta = {
        "function": loss_fn_y_delta,
        "lambda": lambda_y_delta,
        "options": None
    }


    lambda_y_reg_extra = config.get('loss_weights', {}).get('lambda_y_reg_extra', 0.0)
    loss_fn_y_reg_extra = nn.MSELoss()
    if lambda_y_reg_extra < 1.0e-8: 
        loss_fn_y_reg_extra = dummy
    loss_dct_y_reg_extra = {
        "function": loss_fn_y_reg_extra,
        "lambda": lambda_y_reg_extra,
        "options": None
    }

    r = {
        "x": loss_dct_x,
        #
        "y": loss_dct_y,
        "y_delta": loss_dct_y_delta,
        "y_reg_extra": loss_dct_y_reg_extra,
        # 
        "z": loss_dct_z,
        #
        "y_delta": loss_dct_y_delta,
        "y_reg_extra": loss_dct_y_reg_extra,
        #
        "z_triplet_classes": loss_dct_z_triplet_classes,
        "z_triplet_WST_score": loss_dct_z_triplet_WST_score, 
        # "z_triplet_WST_time": loss_dct_z_triplet_WST_time,   # Later ... or not 
        #
        "z_score_consistency_regularizer": loss_dct_z_score_consistency_regularizer
    }

    return r
