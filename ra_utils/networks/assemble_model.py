import ra_utils.networks.architecture
from ra_utils.networks.architecture import (
    ResNet18Encoder,
    ResNet34Encoder,
    ResNet50Encoder,
    make_mlp,
    EncoderClassifierNetwork,
    MultiModalImageScoreTypeNetwork,
    ROI_type_encoder,
    model_interface_forward
)
import numpy as np
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
from ra_utils.utils.utils_SHS_scoring import get_classes



def make_params_a_la_Paul(config):
    params = {"chosen_score": config["data"]["scores"]}
    params = {**params, **config["model_params"]}

    if params["binary"] == 0:
        # params["n_classes"] = len(unique_tmp)
        # params["classes"] = unique_tmp
        params["n_classes"] = 6
        params["classes"] = np.array([0., 1., 2., 3., 4., 5.])
        if params["chosen_score"] == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
            params["n_classes"] = 26
            params["classes"] = np.arange(26.0)
        # CW: 16 Makes no sense to me (ERO 0, ... 4)
        if params["chosen_score"] == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
            params["n_classes"] = 16
            params["classes"] = np.arange(16.0)
    elif params["binary"] == 1:
        params["n_classes"] = 2
        params["classes"] = np.array([0., 1.])
    elif params["binary"] == 2:
        params["n_classes"] = 2
        params["classes"] = np.array([0., 1.])
    else:
        print("please define \"binary\" ")
    return params

import ra_utils.utils.utils
import pydoc


def import_and_initialize_model(config, model_params):
        model_params = ra_utils.utils.utils.unflatten_dict(config["model_params"])
        model_params = ra_utils.utils.utils.import_kwargs_with_pydoc(model_params, 
                                                                     kwargs_req_import = config.get("model_kw_requires_import", []), 
                                                                     raise_if_not_found=True)
            # deal with list_like_dict
        dct_keys_to_convert_to_lists = config["model_dct_keys_to_convert_to_lists"]
        for k in dct_keys_to_convert_to_lists:
            tmp_list = ra_utils.utils.utils.convert_list_like_dict_to_list(model_params[k])
            model_params[k] = tmp_list
        model_name = config["model_name"]
        model_class = ra_utils.utils.utils.pydoc_locate_targets([model_name], chill=False)[0]
        model = model_class(**model_params)
        return model




def get_model_for_SHS_scoring(config, model_params):
    model_name = config["model_name"]
    if model_name == "AutoscorRA":
        params = make_params_a_la_Paul({'data': {"scores": config["data"]["scores"]},
                                        "model_params": model_params})

        model = Custom_VGG(ipt_size=(128, 128),
                           pretrained=True,
                           num_classes=params["n_classes"],
                           vgg_type=params["vgg_type"],
                           regression=params["regression"],
                           ordinal=params["ordinal"])
        return model
    if model_name == "ResNet18":
        out_dim = model_params["N_classes"]
        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "ResNet34":
        out_dim = model_params["N_classes"]
        encoder = ResNet34Encoder(weights='ResNet34_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "ResNet50":
        out_dim = model_params["N_classes"]
        encoder = ResNet50Encoder(weights='ResNet50_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "MultiModalImageScoreTypeNetwork__ResNet18":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

    if model_name == "MultiModalImageScoreTypeNetwork__ResNet34":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet34Encoder(weights='ResNet34_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

 
    if model_name == "MultiModalImageScoreTypeNetwork__ResNet50":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet50Encoder(weights='ResNet50_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

    else:
        raise ValueError(f"Model {model_name} not implemented yet.")


