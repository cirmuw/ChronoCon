import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional, Literal, List


#--------------------------------------------------------------#
#-------------------------  interfaces-------------------------#
#--------------------------------------------------------------#


def model_interface_forward(model: nn.Module, batch: dict, device="cpu",
                            options: Literal["image only", "image + score_type"] = "image only"):
    if options == "image only":
        X = batch["img"].to(device)
        return model(X)

    elif options == "image + score_type":
        X = batch["img"].to(device)
        score_types = batch["score_type"]   # List of strings (N_batch)
        return model(images=X, score_types=score_types)
    
    else: 
        raise ValueError(f"model_interface_forward :: {option = } not supported ")




#--------------------------------------------------------------#
#-------------------------    utils   -------------------------#
#--------------------------------------------------------------#

def make_mlp(
    latent_dim: int = 320,
    hidden_dim: int = 256,
    normalization: Literal["batch", "instance"] = "batch",
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
        elif normalization == "instance":
            return nn.Sequential(
                nn.Unflatten(1, (1, norm_dim)),  # ->  Nb, 1, hidden_dim
                nn.InstanceNorm1d(1, **norm_op_kwargs),
                nn.Flatten()  # -> Nb, hidden_dim
            )

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