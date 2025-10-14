import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional, Literal, List, Dict

from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet import (
        resnet18,
        resnet34,
        resnet50
    ) 

from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT import (
        resnet18_decoder,
        resnet34_decoder,
        resnet50_decoder
    ) 

from importlib import resources
import pandas as pd

#--------------------------------------------------------------#
#-------------------------  interfaces-------------------------#
#--------------------------------------------------------------#


# def model_interface_forward(model: nn.Module, batch: dict, device="cpu",
#                             options: Literal["image only", "image + score_type"] = "image only"):
#     if options == "image only":
#         X = batch["img"].to(device)
#         return model(X)

#     elif options == "image + score_type":
#         X = batch["img"].to(device)
#         score_types = batch["score_type"]   # List of strings (N_batch)
#         return model(images=X, score_types=score_types)
    
#     else: 
#         raise ValueError(f"model_interface_forward :: {option = } not supported ")


def model_interface_forward(model: nn.Module, batch: dict, device="cuda",
                            options: Literal["IN: image; OUT: class_logits", 
                                             "IN: image + score_type; OUT: class_logits",
                                             "IN: image; OUT: recon, latent",
                                             "IN: image + score_type; OUT: recon, latent",                                             
                                             ] = "IN: image only; OUT: class_logits"):
    if options == "IN: image; OUT: class_logits":
        X = batch["img"].to(device)
        class_logits =  model(X)
        return {"class_logits": class_logits}

    elif options == "IN: image + score_type; OUT: class_logits":
        X = batch["img"].to(device)
        score_types = batch["score_type"]   # List of strings (N_batch)
        class_logits =  model(images=X, score_types=score_types)
        return {"class_logits": class_logits}
    
    if options == "IN: image; OUT: recon, latent":
        X = batch["img"].to(device)
        recon, z =  model(X)
        return {"recon": recon, "latent": z}    
    
    else: 
        raise ValueError(f"model_interface_forward :: {options = } not supported ")







#--------------------------------------------------------------#
#-------------------------    utils   -------------------------#
#--------------------------------------------------------------#

def make_mlp(
    latent_dim: int = 320,
    hidden_dim: int = 256,
    normalization: Literal["batch", "instance", "layer"] = "batch",
    norm_op_kwargs: dict = {},
    depth: int = 2,
    nonlin=nn.ReLU,
    nonlin_kwargs: dict = {},
    dropout_op: Optional[nn.Module] = nn.Dropout,
    dropout_op_kwargs: dict = {"p": 0.2},
    out_dim=1
):
    def get_normalization_block(norm_dim: int = hidden_dim):
        if normalization == "batch":
            return nn.BatchNorm1d(norm_dim, **norm_op_kwargs)
        elif normalization == "layer":
            return nn.LayerNorm(norm_dim, **norm_op_kwargs)
        elif normalization == "instance":
            return nn.Sequential(
                nn.Unflatten(1, (1, norm_dim)),  # ->  Nb, 1, hidden_dim
                nn.InstanceNorm1d(1, **norm_op_kwargs),
                nn.Flatten()  # -> Nb, hidden_dim
            )
        else: 
            raise NotADirectoryError(f"{normalization = }")

    mlp_layers = []

    # note that there was already a linear layer just before that in the reduction step # CW note true!!
    if depth >0:
        if dropout_op is not None:
            mlp_layers.append(dropout_op(**dropout_op_kwargs))
        mlp_layers.append(get_normalization_block(latent_dim))
        mlp_layers.append(nonlin(**nonlin_kwargs))

    if depth > 1:
        mlp_layers.append(nn.Linear(latent_dim, hidden_dim))
        if dropout_op is not None:
            mlp_layers.append(dropout_op(**dropout_op_kwargs))
        mlp_layers.append(get_normalization_block(hidden_dim))
        mlp_layers.append(nonlin(**nonlin_kwargs))

        for i in range(depth-2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout_op is not None:
                mlp_layers.append(dropout_op(**dropout_op_kwargs))
            mlp_layers.append(get_normalization_block(hidden_dim))
            mlp_layers.append(nonlin(**nonlin_kwargs))

        mlp_layers.append(nn.Linear(hidden_dim, out_dim))

    else:
        mlp_layers.append(nn.Linear(latent_dim, out_dim))

    return nn.Sequential(*mlp_layers)

#--------------------------------------------------------------#
#-------------------------    Classes   -----------------------#
#--------------------------------------------------------------#


class TrivialReturnInput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


#------------------ Image encoder -----------------------------#

class ResNet18Encoder(nn.Module):
    def __init__(self, **resenet_kwargs):
        """
        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        """
        
        super().__init__()
        self.net = torchvision.models.resnet18(**resenet_kwargs)
        self.out_features = self.net.fc.in_features  # in features of last layer 
        self.net.fc = TrivialReturnInput() # remove last layer of 
        
    def forward(self, x):
        z = self.net(x)
        return z
        


class ResNet34Encoder(nn.Module):
    def __init__(self, **resenet_kwargs):
        """
        encoder = ResNet34Encoder(weights='ResNet18_Weights.DEFAULT')
        """
        
        super().__init__()
        self.net = torchvision.models.resnet34(**resenet_kwargs)
        self.out_features = self.net.fc.in_features  # in features of last layer 
        self.net.fc = TrivialReturnInput() # remove last layer of 
        
    def forward(self, x):
        z = self.net(x)
        return z
        


class ResNet50Encoder(nn.Module):
    def __init__(self, **resenet_kwargs):
        """
        encoder = ResNet50Encoder(weights='ResNet50_Weights.DEFAULT')
        """
        
        super().__init__()
        self.net = torchvision.models.resnet50(**resenet_kwargs)
        self.out_features = self.net.fc.in_features  # in features of last layer 
        self.net.fc = TrivialReturnInput() # remove last layer of 
        
    def forward(self, x):
        z = self.net(x)
        return z
        
                    
#--------------------------------------------------------------#
#------------------------  Interfaces   -----------------------#
#--------------------------------------------------------------#
        
class EncoderClassifierNetwork(nn.Module):
    def __init__(self,
                 encoder,
                 classifier,
                 return_latent_representation = False,
                 preprocessor = None
                 ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.return_latent_representation = return_latent_representation
        self.preprocessor = preprocessor
        
    def forward(self, x):
        if self.preprocessor == None:
            x_preprocessed = x
        else:
            x_preprocessed = self.preprocessor(x)
        
        z = self.encoder(x_preprocessed)
        y = self.classifier(z)
        if self.return_latent_representation:
            return y, z
        else:
            return y
    


        
class ROI_type_encoder(nn.Module):
    def __init__(self, classes: List[str], out_dim: Optional[int] = None, normalized: bool = False):
        """
        Args:
            classes: List of class labels.
            out_dim: Desired output dimension. If None, the one-hot encoded vector (len(classes)) is used.
            normalized: Whether to L2-normalize the output.
        """
        super().__init__()
        self.classes = classes
        self.class2idx = {cls: idx for idx, cls in enumerate(classes)}
        # Use output_dim to consistently define the encoder's output size.
        self.output_dim = out_dim if out_dim is not None else len(classes)
        self.normalized = normalized

        # Create a linear layer only if a specific output dimension is provided.
        if out_dim is not None:
            self.linear1 = torch.nn.Linear(len(classes), self.output_dim, dtype=torch.float32)
        else:
            self.linear1 = None

    def forward_batch(self, score_types: List[str], device=None) -> torch.Tensor:
        """
        Vectorized forward pass: converts a list of string labels to one-hot vectors and optionally
        applies a linear transformation followed by normalization.
        
        Args:
            score_types: List of score type strings.
            device: Optional; if not provided, defaults to the device of the module's parameters.
        
        Returns:
            Tensor of shape (N, output_dim)
        """
        # Automatically determine device if not provided.
        if device is None:
            device = next(self.parameters()).device


        # Maybe performace bottleneck -> Can be put into dataloader if needed ....
        idx_list = [self.class2idx[st] for st in score_types]
        idx_tensor = torch.tensor(idx_list, device=device)
        
        x = F.one_hot(idx_tensor, num_classes=len(self.classes)).float()
        if self.linear1 is not None:
            x = self.linear1(x)
        if self.normalized:
            x = F.normalize(x, p=2, dim=-1)
        
        return x

    def forward(self, x: str):
        """
        Single-sample forward for convenience. In typical use, you'll want to use `forward_batch`.
        
        Args:
            x: A single score type string.
            
        Returns:
            Tensor of shape (output_dim,)
        """
        return self.forward_batch([x])[0]


class MultiModalImageScoreTypeNetwork(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        score_type_encoder: nn.Module,
        classifier: nn.Module,
        return_latent_representation: bool = False
    ):
        """
        :param image_encoder: e.g. your ResNet18Encoder (outputs a latent of shape [N, D_img])
        :param score_type_encoder: e.g. your ROI_type_encoder (outputs a latent of shape [D_tab])
                                   must handle single or batch of strings
        :param classifier: some MLP or linear head that takes [D_img + D_tab] -> #classes
        :param return_latent_representation: whether to return (prediction, full_concat_latent)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.score_type_encoder = score_type_encoder
        self.classifier = classifier
        self.return_latent_representation = return_latent_representation

    def forward(self, images: torch.Tensor, score_types: List[str]):
        """
        :param images: tensor of shape [N, 3, H, W]
        :param score_types: list of strings (length N)
        """
        z_img = self.image_encoder(images)  # shape [N, D_img]
        z_tab = self.score_type_encoder.forward_batch(score_types, device=images.device)  # shape [D_tab]
 
        # Concatenate them
        z_concat = torch.cat([z_img, z_tab], dim=1)  # shape [N, D_img + D_tab]

        # Classify
        logits = self.classifier(z_concat)  # shape [N, #classes]

        if self.return_latent_representation:
            return logits, z_concat
        else:
            return logits
        

class MultiModalImageScoreTypeNetworkAE(nn.Module):
    def __init__(
        self,
        autoencoder: nn.Module, # return X_recon, z
        score_type_encoder: Optional[nn.Module] = None,
        preprocessor: Optional[nn.Module] = None,
        postprocessor: Optional[nn.Module] = None
    ):
        """Wrapper to add score type and keep interface the same: 
           (recon, z_prime)  <-- img

        Args:
            autoencoder (nn.Module): (recon, z)  <-- img
            score_type_encoder (nn.Module): 
        forward: 
            returns (recon, z_prime)
        """


        super().__init__()
        self.auto_encoder = autoencoder
        self.score_type_encoder = score_type_encoder
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        if score_type_encoder == None: 
            latent_dim_tab = 0
        else: 
            latent_dim_tab = score_type_encoder.output_dim
        latent_dim_autoencoder = autoencoder.latent_dim 
        self.latent_dim = latent_dim_autoencoder + latent_dim_tab


    def forward(self, images: torch.Tensor, score_types: List[str]):
        """
        :param images: tensor of shape [N, 3, H, W]
        :param score_types: list of strings (length N)
        """

        if self.preprocessor == None:
            x_preprocessed = images
        else:
            x_preprocessed = self.preprocessor(images)

        x_recon, z_img = self.auto_encoder(x_preprocessed, score_types)  # shape [N, D_img]
        if self.postprocessor != None:
            x_recon = self.postprocessor(x_recon)



        if self.score_type_encoder == None: 
            z_concat = z_img
        else: 
            z_tab = self.score_type_encoder.forward_batch(score_types, device=images.device)  # shape [D_tab]
            z_concat = torch.cat([z_img, z_tab], dim=1)  # shape [N, D_img + D_tab]
        return x_recon, z_concat
        

        


def add_preprocessor_postprocessor_roi_type_encoder_to_model_AE(model_AE,
                                                                config: dict):
    # preprocessor
    name = config["model"].get("preprocessor", {}).get("name", None)
    params = config["model"].get("preprocessor", {}).get("params", {})
    if name == None: 
        preprocessor = None
    elif name == "ResNetRGBMaker":
        preprocessor = PreprocessingResNetRGBMaker(**params)
    else: 
        raise NotImplementedError(f"{name =}")
    
    # postprocessor_recon: 
    name = config["model"].get("postprocessor_recon", {}).get("name", None)
    params = config["model"].get("postprocessor_recon", {}).get("params", {})
    if name == None: 
        postprocessor_recon = None
    elif name == "PostprocessingGrayScaleMaker_v2":
        postprocessor_recon = PostprocessingGrayScaleMaker_v2(**params)
    elif name == "PostprocessingGrayScaleMaker_v3":
        postprocessor_recon = PostprocessingGrayScaleMaker_v3(**params)
    elif name == "PostprocessingGrayScaleMaker":
        postprocessor_recon = PostprocessingGrayScaleMaker(**params)
    else: 
        raise NotImplementedError(f"{name =}")    

    # score_type_encoder

    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
        df_scores_meta = pd.read_csv(f)
    score_groups_list = sorted(set(list(df_scores_meta["score_name"])))
    #score_groups_list = []
    #for k, v in config["data"]["score_groups"].items():
    #    for s in v: 
    #        score_groups_list.extend(v)
    name = config["model"].get("score_type_encoder", {}).get("name", None)
    params = config["model"].get("score_type_encoder", {}).get("params", {})
    if name == None: 
        score_type_encoder = None
    elif name == "ROI_type_encoder":
        score_type_encoder = ROI_type_encoder(classes=score_groups_list, **params)
    else: 
        raise NotImplementedError(f"{name =}")    
    
    model_AE_mod = MultiModalImageScoreTypeNetworkAE(model_AE, 
                                                    score_type_encoder=score_type_encoder,
                                                    preprocessor=preprocessor, 
                                                    postprocessor=postprocessor_recon)
    return model_AE_mod



# ---------------------------------------------------------------------------
# multi‑head classifier model and helper functions
# ---------------------------------------------------------------------------



def make_score_type_2_head_name_dct(classifier_head_infos: dict):
    out = {}
    used_score_types = set()
    for head_name, v in classifier_head_infos.items():
        st = set(v["score_types"])
        dup = st & used_score_types
        assert not dup, f"duplicate score type(s) {dup} in head '{head_name}'"
        for s in st:
            out[s] = head_name
        used_score_types |= st
    return out

# classifier_head_infos = {
#     "PIP_2-5_EP": {
#         "out_dim": 6,
#         "score_types":   ['PIPIIEP', 'PIPIIIEP', 'PIPIVEP', 'PIPVEP'],
#         "loss_weight": 1.0 
#     },
#     "PIP_1_EP": {
#         "out_dim": 6,
#         "score_types":   ['IPIEP'],
#         "loss_weight": 1.0 
#     }
# }


class ClassifierHeads(nn.Module):
    def __init__(self,
                 classifier_head_infos, # = classifier_head_infos,  
                 latent_dim: int = 480,
                 mlp_kwargs = {"depth": 0, 
                               "dropout_op": None}):
        super(ClassifierHeads, self).__init__()
        self.classifier_head_infos = classifier_head_infos.copy()
        
        # input checks on classifier_head_infos -> No overlaps in score_types
        self.score_type_2_head_name = make_score_type_2_head_name_dct(classifier_head_infos)
        self.heads = nn.ModuleDict({k: make_mlp(**{"latent_dim": latent_dim, 
                                                 "out_dim": v["out_dim"]},  
                                                 **mlp_kwargs) for k,v in classifier_head_infos.items()})
        
    def forward(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        score_types: List[str],
        *,
        return_dict: bool = False,  # <-- NEW
    ) -> (
        Dict[str, tuple[torch.Tensor, torch.Tensor]]
        | torch.Tensor
    ):
        """
        Parameters
        ----------
        z            : latent vectors, shape [B, latent_dim]
        score_types  : length-B list with one entry per sample
        return_dict  : if True  → always return a dict
                       if False → keep the old behaviour
        Returns
        -------
        * return_dict = True
            {head_name: (idx, logits)}  where
                idx    : 1-D LongTensor with the sample positions that
                         belong to this head, length n_h
                logits : Tensor [n_h, out_dim_h]
        * return_dict = False
            Same tensor output as before (fast path when all samples share
            one head, or padded tensor+mask when heads differ).
        """
        if len(z) != len(score_types):
            raise ValueError(
                f"z has {len(z)} rows but score_types has {len(score_types)} items."
            )

        active_heads = [self.score_type_2_head_name[s] for s in score_types]

        # ------------------------------------------------------------------ #
        # ── 1. Return the new dict format, used when return_dict = True ─────
        # ------------------------------------------------------------------ #
        if return_dict:
            out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for head_name, head in self.heads.items():        # ← iterate over *all* heads
                # positions in the batch that belong to this head
                idx = torch.as_tensor(
                    [i for i, h in enumerate(active_heads) if h == head_name],
                    dtype=torch.long,
                    device=z.device,
                )

                if idx.numel():                               # ≥1 example for this head
                    logits = head(z[idx])                     # [n_h, out_dim_h]
                else:                                         # no sample ➜ return empties
                    out_dim = head[-1].out_features
                    logits = torch.empty((0, out_dim), device=z.device, dtype=z.dtype)  
                out[head_name] = (idx, logits)                # always present

            return out

        # ------------------------------------------------------------------ #
        # ── 2. Legacy tensor output  (unchanged) ────────────────────────────
        # ------------------------------------------------------------------ #
        if len(set(active_heads)) == 1:          # fast path
            return self.heads[active_heads[0]](z)

        # different out_dim per head → pad to the largest width
        max_dim = max(self.heads[h][-1].out_features for h in set(active_heads))
        logits = z.new_zeros(len(z), max_dim)
        mask   = torch.zeros_like(logits, dtype=torch.bool)

        for head in set(active_heads):
            idx   = [i for i, h in enumerate(active_heads) if h == head]
            out_h = self.heads[head](z[idx])          # [n_h, out_dim_h]
            logits[idx, : out_h.shape[1]] = out_h
            mask[idx, : out_h.shape[1]]   = True      # valid columns

        return logits, mask






class ContinuesScoreEsimator(nn.Module):
    def __init__(self,
                 #df_scores_meta,
                 task_type_y:  Literal['classification', 'regression', 'classification_regression_mix'] = "classification_regression_mix",
                 bias=True):
        super(ContinuesScoreEsimator, self).__init__()


        with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
            df_scores_meta = pd.read_csv(f)

        module_dct = {}
        for index, row in df_scores_meta.iterrows():
            if task_type_y == "classification":
                N_in = row['limit'] + 1 #  1 is for score 0
            elif task_type_y == "regression":
                N_in = 1 # Does not make much sense to use in this case...
            elif task_type_y == "classification_regression_mix":
                N_in = row['limit'] + 2 #  1 is for score 0  one is for extra regression part
            else: 
                raise ValueError(f"{task_type_y = }")                
            module_dct[f"{row['score_name']}"] = nn.Linear(N_in, 1, bias=bias)

        self.score_estimators = nn.ModuleDict(module_dct)
        self.task_type_y = task_type_y


    def forward(
        self,
        x_input, # [B, Nin_] # N_in can differ 
        score_types: List[str],     # [B]
    ):

        assert set(score_types) - set(self.score_estimators.keys()) == set(), f"{set(score_types) - set(self.score_estimators.keys())  = }"

        if x_input.shape[0] != len(score_types):
            raise ValueError(
                f" {x_input.shape[0] = }  {len(score_types) = } "
            )

        x = x_input.clone()
        if self.task_type_y == "classification":
            x = nn.functional.softmax(x, dim=1)
        if self.task_type_y == "classification_regression_mix":
            x[..., 1:] = nn.functional.softmax(x[..., 1:], dim=1) # First is regression part -> rest is logits. 
        
        B, _ = x.shape
        out = torch.zeros((B), dtype=x_input.dtype, device=x_input.device)
        # score_types_np = np.array(score_types)

        for head_name, head in self.score_estimators.items():        # ← iterate over *all* heads
            idx = torch.as_tensor(
                [i for i, h in enumerate(score_types) if h == head_name],
                dtype=torch.long,
                device=x_input.device,
            )

            if idx.numel():
                out[idx] = (head(x[idx])).squeeze()
        return out



    # def __make_mlp(self, in_dim, out_dim, hidden_dim=256, dropout_p=None):
    #     layers = [nn.Linear(in_dim, hidden_dim),
    #           #nn.BatchNorm1d(hidden_dim),
    #           nn.LayerNorm(hidden_dim),
    #           nn.ReLU(inplace=True)]
    #     if dropout_p is not None:
    #         layers.append(nn.Dropout(p=dropout_p, inplace=True))
    #     layers.append(nn.Linear(hidden_dim, out_dim))
    #     return nn.Sequential(*layers)
        
class EncoderDecoderNetwork(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 preprocessor = None,
                 postprocessor = None, 
                 reducer = None
                 ):
        super().__init__()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.reducer = reducer
        
        
    def forward(self, x, *args, **kwargs):  # Other model is called with signature (img, score_types)
        if self.preprocessor == None:
            x_preprocessed = x
        else:
            x_preprocessed = self.preprocessor(x)

        z = self.encoder(x_preprocessed)
        x_pred = self.decoder(z)

        if self.postprocessor != None:
            x_pred = self.postprocessor(x_pred)

        if self.reducer != None: 
            z = self.reducer(z)


        return x_pred, z

# class PreprocessingResNetRGBMaker(nn.Module):
#     """
#     input  dim (Nb, Nc=1, ...)
#     output dim (Nb, Nc=3, ...)
#     can also normalize channels (which pretrained resnet expects)
#     """
        
#     def __init__(self, 
#                  mean = (0.485, 0.456, 0.406),
#                  std = (0.229, 0.224, 0.225)
#                  ):
#         super().__init__()
#         self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
                    
#     def forward(self, x):
#         # Repeat the single channel to create 3 channels
#         x = x.repeat(1, 3, 1, 1)
#         # Normalize the channels
#         x = self.normalize(x)
#         return x


# Maybe add 01 normalization
class PreprocessingResNetRGBMaker(nn.Module):
    """
    Input  : (N, C, H, W)   [C may be 1 or 3]
    Output : (N, 3, H, W)   [RGB, ImageNet-normalized]
    """
    def __init__(
        self,
        enforce_01_normalization: bool = False,
        mean = (0.485, 0.456, 0.406),
        std  = (0.229, 0.224, 0.225),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.enforce_01_normalization = enforce_01_normalization
        self.eps = eps
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def _minmax_01(self, x: torch.Tensor) -> torch.Tensor:
        # Per-sample (and per-channel) min–max over spatial dims
        # x: (N, C, H, W)
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        denom = (x_max - x_min).clamp_min(self.eps)
        x = (x - x_min) / denom
        return x.clamp_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure float
        x = x.float()

        if self.enforce_01_normalization:
            x = self._minmax_01(x)

        # Repeat to 3 channels if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # ImageNet normalization for ResNet backbones
        x = self.normalize(x)
        return x
    


class PostprocessingGrayScaleMaker(nn.Module):
    """
    input  dim (N, 3, H, W)
    output dim (N, 1, H, W)
    """
    def __init__(self, 
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), 
                 output_function : Literal["None", "relu", "sigmoid"] = "None"
                 ):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
        self.output_function_str = output_function
        if output_function == "None":
            self.output_function = None
        elif output_function == "relu":
            self.output_function = nn.ReLU()
        elif output_function == "sigmoid":
            self.output_function = nn.Sigmoid()
        else: 
            raise ValueError(f"{output_function=}")
                
    def forward(self, x):
        # Unnormalize the channels
        x = x * self.std.to(x.device) + self.mean.to(x.device)
        # Average over the channels
        x = torch.mean(x, dim=1, keepdim=True)
        if self.output_function != None:
            x = self.output_function(x)
        return x

class PostprocessingGrayScaleMaker_v2(nn.Module):
    """
    input  dim (N, 3, H, W)
    output dim (N, 1, H, W)
    """
    def __init__(self, output_function: Literal["sigmoid", "relu", "None"] = "sigmoid"):
        super().__init__()
        self.output_function_str = output_function
        if output_function == "None":
            self.output_function = None
        elif output_function == "relu":
            self.output_function = nn.ReLU()
        elif output_function == "sigmoid":
            self.output_function = nn.Sigmoid()
        else: 
            raise ValueError(f"{output_function=}")
                
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=False)
        x.unsqueeze_(1)
        if self.output_function != None:
            x = self.output_function(x)
        return x
    
class PostprocessingGrayScaleMaker_v3(nn.Module):
    """
    simply use first dim
    input  dim (N, X, H, W)
    output dim (N, 1, H, W)
    """
    def __init__(self, output_function: Literal["sigmoid", "relu", "None"] = "sigmoid"):
        super().__init__()
        self.output_function_str = output_function
        if output_function == "None":
            self.output_function = None
        elif output_function == "relu":
            self.output_function = nn.ReLU()
        elif output_function == "sigmoid":
            self.output_function = nn.Sigmoid()
        else: 
            raise ValueError(f"{output_function=}")
                
    def forward(self, x):
        x = x[:, 0:1, ...]
        if self.output_function != None:
            x = self.output_function(x)
        return x    

def build_ResNetAutoEncoder_v2(arch="resnet18", output_function="sigmoid"):
    out_ch = 3
    if arch == 'resnet18':
        encoder = resnet18(pretrained=True)
        decoder = resnet18_decoder(out_ch=out_ch)
    elif arch == 'resnet34':
        encoder = resnet34(pretrained=True)
        decoder = resnet34_decoder(out_ch=out_ch)
    elif arch == 'resnet50':
        encoder = resnet50(pretrained=True)
        decoder = resnet50_decoder(out_ch=out_ch)
    else:
        raise NotImplementedError

    reducer = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1)])
    preprocessor = PreprocessingResNetRGBMaker()
    #postprocessor = PostprocessingGrayScaleMaker(output_function="sigmoid")
    postprocessor = PostprocessingGrayScaleMaker_v2(output_function=output_function)


    net = EncoderDecoderNetwork(encoder=encoder,
                                decoder=decoder,  
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                reducer=reducer)
    return net



def build_ResNetAutoEncoder_v2p1(arch="resnet18", output_function="sigmoid"):
    from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_using_basic_block_decoder import (
        resnet18_decoder_v2
    )
    if arch == 'resnet18':
        encoder = resnet18(pretrained=True)
        decoder = resnet18_decoder_v2()
    else:
        raise NotImplementedError

    reducer = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1)])
    preprocessor = PreprocessingResNetRGBMaker()
    #postprocessor = PostprocessingGrayScaleMaker(output_function="sigmoid")
    postprocessor = PostprocessingGrayScaleMaker_v2(output_function=output_function)


    net = EncoderDecoderNetwork(encoder=encoder,
                                decoder=decoder,  
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                reducer=reducer)
    return net
    
class ResNetAutoEncoder(nn.Module):
    def __init__(self, arch='resnet18'):
        super().__init__()
        out_ch=3
        if arch == 'resnet18':
            self.encoder = resnet18(pretrained=True)
            self.decoder = resnet18_decoder(out_ch=out_ch)
        elif arch == 'resnet34':
            self.encoder = resnet34(pretrained=True)
            self.decoder = resnet34_decoder(out_ch=out_ch)
        elif arch == 'resnet50':
            self.encoder = resnet50(pretrained=True)
            self.decoder = resnet50_decoder(out_ch=out_ch)
        else:
            raise NotImplementedError
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.latend_dim = self.encoder.fc.in_features

    def forward(self, x): # , *args, **kwargs):  # Other model is called with signature (img, score_types)
        z = self.encoder(x)
        z_out = self.avgpool(z)
        z_out = torch.flatten(z_out, 1)
        
        # Option 1: recon before red. 
        x_hat = self.decoder(z)
        
        # Option 2: recon after red. 
        #x_hat = self.decoder(z_out.unsqueeze(2).unsqueeze(3))  # Then dims will not match... 
        
        return x_hat, z_out
    
# Just to use the same interface with recon, ... also for this model
class ResNetNOAutoEncoder(nn.Module):
    def __init__(self, arch='resnet18'):
        super().__init__()
        if arch == 'resnet18':
            self.encoder = resnet18(pretrained=True)
        elif arch == 'resnet34':
            self.encoder = resnet34(pretrained=True)
        elif arch == 'resnet50':
            self.encoder = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_dim = self.encoder.latent_dim

    def forward(self, x, *args, **kwargs): # added dummy arguments to allow for same interface
        z = self.encoder(x)
        z_out = self.avgpool(z)
        z_out = torch.flatten(z_out, 1)
        x_hat = x*0.0 
        return x_hat, z_out
    
class DummyReturnZeroLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=self.device)

class DummyReturnZeroLossAndDict(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=self.device), {}



class DummyReturnZeroLossMulti(nn.Module):
    def __init__(self, device="cuda", size=2):
        super().__init__()
        self.device = device
        self.size = size
    
    def forward(self, *args, **kwargs):
        if isinstance(self.size, int):
            # Return a tuple of tensors for unpacking
            return tuple(torch.tensor(0.0, device=self.device) for _ in range(self.size))
        else:
            # Return a tensor with the specified size
            return torch.zeros(self.size, device=self.device)
    
    def __repr__(self):
        return f"DummyReturnZeroLossMulti(device='{self.device}', size={self.size})"