import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
import torch
import matplotlib.cm as cm

# --- pick a reasonable default target layer if you don't want to pass one ---
def find_last_conv(module: torch.nn.Module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
            name_last = name
    if last is None:
        raise ValueError("No Conv2d layer found in model_AE. Pass target_layer manually.")
    print("Used layer: ", name_last)
    return last

class GradCAM:
    def __init__(self, model_ae, model_c, target_layer=None):
        self.model_ae = model_ae
        self.model_c  = model_c
        self.target_layer = target_layer or find_last_conv(model_ae)

        self.activations = None
        self.gradients   = None

        # hooks
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # out shape: [B, C, H', W']
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] shape: [B, C, H', W']
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def _upsample_and_normalize(self, cam, H, W):
        cam = F.interpolate(cam.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        # normalize each item to [0,1]
        cam_min = cam.amin(dim=(1,2), keepdim=True)
        cam_max = cam.amax(dim=(1,2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam

    def __call__(self, X, score_type, target_class: int | None = None):
        """
        X: tensor [1, C, H, W]
        score_type: whatever your models expect (e.g. ['H_JSN_PIPIII'])
        target_class: optional class index; default = argmax
        returns: cam [1, H, W] in [0,1]
        """
        # ensure clean graph
        self.model_ae.zero_grad(set_to_none=True)
        self.model_c.zero_grad(set_to_none=True)

        # forward (no torch.no_grad here; we need a graph!)
        X_recon, z = self.model_ae(X, score_type)  # z must NOT be detached inside model_AE
        logits = self.model_c(z, score_type)       # shape [1, K] for the active head

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # backpropagate class score
        score = logits[:, target_class]            # shape [1]
        score.backward()

        # compute Grad-CAM
        A = self.activations           # [1, C, H', W']
        dYdA = self.gradients          # [1, C, H', W']
        weights = dYdA.mean(dim=(2, 3), keepdim=True)     # GAP over spatial dims -> [1, C, 1, 1]
        cam = torch.relu((weights * A).sum(dim=1))        # [1, H', W']

        # upsample to input size and normalize [0,1]
        _, _, H, W = X.shape
        cam = self._upsample_and_normalize(cam, H, W)     # [1, H, W]
        return cam, logits, target_class

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

# --------- convenience for overlay (optional) ----------
def overlay_cam_on_image(img: torch.Tensor, cam: torch.Tensor, alpha: float = 0.4):
    """
    img: [1,1,H,W] or [1,3,H,W] in arbitrary range -> normalized to [0,1] for display
    cam: [1,H,W] in [0,1]
    returns: numpy array [H,W,3] in [0,1]
    """
    img_ = img.detach().cpu().float()
    if img_.shape[1] == 1:
        img_ = img_.repeat(1, 3, 1, 1)
    # scale image to [0,1] for visualization
    imin = img_.amin(dim=(1,2,3), keepdim=True)
    imax = img_.amax(dim=(1,2,3), keepdim=True)
    img_ = (img_ - imin) / (imax - imin + 1e-6)

    cam_ = cam.detach().cpu().float()
    cam_ = cam_.clamp(0, 1)
    cam_rgb = cam_.repeat(1, 3, 1, 1)

    overlay = (1 - alpha) * img_ + alpha * cam_rgb
    return overlay.squeeze(0).permute(1, 2, 0).numpy()



def overlay_cam_heatmap(
    img: torch.Tensor,
    cam: torch.Tensor,
    *,
    alpha: float = 0.45,
    cmap: str = "turbo",          # "turbo" (preferred) or "jet" (classic; red=high)
    use_cam_as_alpha: bool = True,
    gamma: float = 1.0,           # >1 darkens low activations; <1 boosts them
):
    """
    img: [1,1,H,W] or [1,3,H,W], arbitrary range -> internally normalized to [0,1] for display
    cam: [1,H,W] in [0,1]
    returns: np.ndarray [H,W,3] in [0,1]
    """
    # --- prep image (to [0,1], keep grayscale look) ---
    img_ = img.detach().cpu().float()
    if img_.shape[1] == 1:
        img_ = img_.repeat(1, 3, 1, 1)
    imin = img_.amin(dim=(1,2,3), keepdim=True)
    imax = img_.amax(dim=(1,2,3), keepdim=True)
    img01 = (img_ - imin) / (imax - imin + 1e-6)           # [1,3,H,W]
    img_np = img01.squeeze(0).permute(1, 2, 0).numpy()      # [H,W,3]

    # --- prep CAM -> color heatmap ---
    cam_ = cam.detach().cpu().float().clamp(0, 1).squeeze(0)  # [H,W]
    if gamma != 1.0:
        cam_ = cam_.pow(gamma)
    try:
        cmap_fn = cm.get_cmap(cmap)
    except ValueError:
        cmap_fn = cm.get_cmap("jet")  # fallback
    heatmap = cmap_fn(cam_.numpy())[..., :3]                # [H,W,3], in [0,1]

    # --- blend ---
    if use_cam_as_alpha:
        alpha_map = (alpha * cam_.numpy())[..., None]       # stronger where CAM is high
    else:
        alpha_map = alpha
    overlay = (1 - alpha_map) * img_np + alpha_map * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# ----------------- example usage -----------------
# model_AE.eval(); model_c.eval()  # ok for BN/Dropout; Grad-CAM uses gradients from the logit
# X = batch["img"].to(device)  # shape [1,C,H,W]
# score_type = batch["score_type"]

# 1) pick a target layer (ideally the last conv of your encoder)
#    If your AE has a ResNet backbone, you might use e.g.: model_AE.encoder.layer4[-1].conv3
#    Otherwise, rely on auto-detection:
#    target_layer = find_last_conv(model_AE)
#    or pass a specific one:
#    target_layer = model_AE.shared_conv[-1]  # if you have this attribute

# gradcam = GradCAM(model_AE, model_c, target_layer=None)  # or pass target_layer
# cam, logits, cls = gradcam(X, score_type)  # cam: [1,H,W] in [0,1]
# overlay = overlay_cam_on_image(X, cam, alpha=0.45)

# import matplotlib.pyplot as plt
# plt.figure(); plt.imshow(overlay); plt.axis("off"); plt.title(f"Grad-CAM (class {cls})"); plt.show()
# gradcam.close()