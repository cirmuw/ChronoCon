import os
from pathlib import Path
import torch
import torch.nn as nn
import ra_utils.utils.utils

# from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
# )

from ra_utils.progressionlearning.models.builder import (
    build_MTANAE, 
    build_MTANAE_v2
)

from ra_utils.networks.architecture import (
    add_preprocessor_postprocessor_roi_type_encoder_to_model_AE,
    ClassifierHeads, 
    ResNetAutoEncoder, 
    ResNetNOAutoEncoder,
    build_ResNetAutoEncoder_v2,
    build_ResNetAutoEncoder_v2p1
)

import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def build_models_AE(model_name: str, config: dict, classifier_head_infos: dict):
    if model_name == "UNetMTANAE + MultiHeadClassifier":
        model_AE = build_MTANAE(in_channels=1, out_channels=1)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=480,  mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = ResNetAutoEncoder(**AE_kwargs)
        latend_dim = model_AE.encoder.fc.in_features  # hack (fc is actually never called)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE_v2 + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = build_ResNetAutoEncoder_v2(**AE_kwargs)  
        latend_dim = model_AE.encoder.fc.in_features  # hack (fc is actually never called)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE_v2p1 + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = build_ResNetAutoEncoder_v2p1(**AE_kwargs)  
        latend_dim = model_AE.encoder.fc.in_features  # hack (fc is actually never called)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetNoAE + MultiHeadClassifier":
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = ResNetNOAutoEncoder(**AE_kwargs)
        latend_dim = model_AE.encoder.fc.in_features  # hack (fc is actually never called)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim,  mlp_kwargs=classifier_kwargs)
    
    elif model_name ==  "UNetMTANAEv2 + MultiHeadClassifier":
        raise NotImplementedError(f"{model_name = }")     

    elif model_name ==  "ResNetMTANAE + MultiHeadClassifier":
        raise NotImplementedError(f"{model_name = }")     
    
    else:
        raise NotImplementedError()
    
    return model_AE, model_c     



def build_models_AE_v2(model_name: str, config: dict, 
                    classifier_head_infos: dict, 
                    attention_paths_dct: dict, 
                    verbose=2):
    if model_name == "UNetMTANAEv2 + MultiHeadClassifier":
        # Encoder / Autoencoder
        model_AE = build_MTANAE_v2(
            u_net_arch="BasicUNet",
            u_net_kwargs=dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
            ),
            attention_paths=attention_paths_dct
            )
        
        # classifier:
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=480,  mlp_kwargs=classifier_kwargs)

    if model_name == "ResNetMTANAE + MultiHeadClassifier":
        # Encoder / Autoencoder
        attention_paths = config["data"]["score_groups"]
        model_kwargs = config["model"]["autoencoder"]
        model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv1(attention_paths=attention_paths,
                                                                                        **model_kwargs)
        # classifier:
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    else:
        raise NotImplementedError(f"{model_name = }")
    if verbose >= 1:
        print("Number of parameters:")
        print(f"   model_AE: {count_parameters(model_AE)/1e6:.2f}e6")
        if verbose >= 2:
            child_params = [(name, count_parameters(module)) for name, module in model_AE.named_children()]
            for name, module_params in sorted(child_params, key=lambda x: x[1], reverse=True):
                print(f"      {name}: {module_params/1e6:.2f}e6")
        print(f"   model_c: {count_parameters(model_c)/1e6:.2f}e6")
    
    return model_AE, model_c



def build_models_AE_v1_and2(model_name: str, config: dict, 
                    classifier_head_infos: dict, 
                    attention_paths_dct: dict = {}, 
                    verbose=2):
    if model_name == "UNetMTANAEv2 + MultiHeadClassifier":
        # Encoder / Autoencoder
        model_AE = build_MTANAE_v2(
            u_net_arch="BasicUNet",
            u_net_kwargs=dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
            ),
            attention_paths=attention_paths_dct
            )
        
        # Not implemented in this model!!!
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        latent_dim = model_AE.latent_dim 

        # classifier:
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latent_dim,  mlp_kwargs=classifier_kwargs)

    elif model_name == "ResNetMTANAE + MultiHeadClassifier":
        # Encoder / Autoencoder
        attention_paths = attention_paths_dct #config["data"]["score_groups"]
        model_kwargs = config["model"]["autoencoder"]
        model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv1(attention_paths=attention_paths,
                                                                                        **model_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)

        # classifier:
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name == "ResNetMTANAEv2 + MultiHeadClassifier":
        # Encoder / Autoencoder
        attention_paths = attention_paths_dct # config["data"]["score_groups"]
        model_kwargs = config["model"]["autoencoder"]
        model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv2(attention_paths=attention_paths,
                                                                                        **model_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        # classifier:
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)


    elif model_name == "ResNetMTANAEv3 + MultiHeadClassifier":
        # Encoder / Autoencoder
        attention_paths = attention_paths_dct # config["data"]["score_groups"]
        model_kwargs = config["model"]["autoencoder"]
        model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv3(attention_paths=attention_paths,
                                                                                        **model_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        # classifier:
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)


    elif model_name == "ResNetMTANAEv4 + MultiHeadClassifier":
        # Encoder / Autoencoder
        attention_paths = attention_paths_dct # config["data"]["score_groups"]
        model_kwargs = config["model"]["autoencoder"]
        model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv4(attention_paths=attention_paths,
                                                                                        **model_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        # classifier:
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)







    # elif model_name == "ResNetMTANAEv2 + MultiHeadClassifier":
    #     # Encoder / Autoencoder
    #     attention_paths = attention_paths_dct # config["data"]["score_groups"]
    #     model_kwargs = config["model"]["autoencoder"]
    #     model_AE = ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan.MTANReconv2(attention_paths=attention_paths,
    #                                                                                     **model_kwargs)
    #     model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
    #     # classifier:
    #     latend_dim = model_AE.latent_dim
    #     cfg = config["model"]["classifier"]
    #     classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
    #                             model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
    #                             model_kw_requires_import=cfg.get("model_kw_requires_import", []))
    #     model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)



    #----------------------------------------------------------#
    #--------------- The ones from v1  ------------------------#
    elif model_name == "UNetMTANAE + MultiHeadClassifier":
        model_AE = build_MTANAE(in_channels=1, out_channels=1)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=model_AE.latent_dim,  mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = ResNetAutoEncoder(**AE_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE_v2 + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = build_ResNetAutoEncoder_v2(**AE_kwargs)  
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetAE_v2p1 + MultiHeadClassifier": 
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = build_ResNetAutoEncoder_v2p1(**AE_kwargs)  
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        latend_dim = model_AE.latent_dim        
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim, mlp_kwargs=classifier_kwargs)

    elif model_name ==  "ResNetNoAE + MultiHeadClassifier":
        AE_kwargs = config["model"]["autoencoder"]
        model_AE = ResNetNOAutoEncoder(**AE_kwargs)
        model_AE = add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE, config)
        latend_dim = model_AE.latent_dim
        cfg = config["model"]["classifier"]
        classifier_kwargs = ra_utils.utils.utils.model_parameter_imports_(cfg["model_params"], 
                                model_dct_keys_to_convert_to_lists=cfg.get("model_dct_keys_to_convert_to_lists", []),
                                model_kw_requires_import=cfg.get("model_kw_requires_import", []))
        model_c = ClassifierHeads(classifier_head_infos=classifier_head_infos, latent_dim=latend_dim,  mlp_kwargs=classifier_kwargs)
    
    else:
        raise NotImplementedError(f"{model_name = }")
    
    if verbose >= 1:
        print("Number of parameters:")
        print(f"   model_AE: {count_parameters(model_AE)/1e6:.2f}e6")
        if verbose >= 2:
            # Level-1 children of model_AE
            child_params = [
                (name, count_parameters(module))
                for name, module in model_AE.named_children()
            ]
            for name, module_params in sorted(child_params, key=lambda x: x[1], reverse=True):
                print(f"      {name}: {module_params/1e6:.2f}e6")

            if verbose >= 3:
                # Level-2 children (grand-children) of model_AE
                grandchild_params = []
                for parent_name, parent_module in model_AE.named_children():
                    for child_name, child_module in parent_module.named_children():
                        full_name = f"{parent_name}.{child_name}"
                        grandchild_params.append((full_name, count_parameters(child_module)))

                for name, params in sorted(grandchild_params, key=lambda x: x[1], reverse=True):
                    print(f"         {name}: {params/1e6:.2f}e6")

        print(f"   model_c:    {count_parameters(model_c)/1e6:.2f}e6")
    
    return model_AE, model_c


