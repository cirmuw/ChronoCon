import torch
import numpy as np
from tqdm import tqdm
import landmarker
from landmarker.heatmap.decoder import heatmap_to_coord

def reconstruct_linear_2D_transform_from_unit_vectors(o_prime, e_x_prime, e_y_prime):
    """
    Input: 
    where o' ... where (0,0) was mapped to 
    where e_x' ... where (1,0) was mapped to
    where e_x' ... where (0,1) was mapped to 
    
    v' = t + A @ v
    v  = A_inv @ (v'-t)
    """
    t = o_prime
    A = np.array([[e_x_prime[0] - t[0], e_y_prime[0] - t[0]], 
                  [e_x_prime[1] - t[1], e_y_prime[1] - t[1]]])
    A_inv = np.linalg.inv(A)
    return t, A, A_inv



def predict_single_image(model, sample, device):
    """
    Predict landmarks for a single image sample.

    Parameters:
        model (torch.nn.Module): Trained model for landmark prediction.
        sample (dict): Dictionary with keys "image", "landmark", "affine", 
                       "dim_original", "spacing", "padding". The "image" 
                       and "landmark" entries are assumed not to include a batch dimension.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (pred_landmark, pred_landmark_transformed, dim_original, pixel_spacing, padding)
               where:
                 - pred_landmark is a torch.Tensor of raw predictions,
                 - pred_landmark_transformed is a numpy.ndarray with transformed landmarks,
                 - the other outputs are tensors with the corresponding metadata.
    """

    model.to(device)
    model.eval()
    with torch.no_grad():
        # Ensure the image has a batch dimension
        image = sample["image"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # Similarly, ensure landmark has a batch dimension.
        landmark = sample["landmark"]
        if landmark.ndim == 1 or landmark.ndim == 2:
            landmark = landmark.unsqueeze(0)
        landmark = landmark.to(device)
        
        dim_orig = sample["dim_original"].to(device)
        pixel_spacing = sample["spacing"].to(device)
        padding = sample["padding"].to(device)

        # Forward pass through the model
        outputs = model(image)
        offset_coords = outputs.shape[1] - landmark.shape[1]
        pred_landmark = heatmap_to_coord(outputs, offset_coords=offset_coords,
                                         method="local_soft_argmax")
        
        # For transformation we assume one image (batch index 0)
        # Extract transformation parameters from the landmark
        o_p, v_ex_p, v_ey_p = landmark[0][:3].cpu().numpy()
        t, A, A_inv = reconstruct_linear_2D_transform_from_unit_vectors(o_p, v_ex_p, v_ey_p)
        t_np = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
        pred_lm_np = pred_landmark[0].cpu().numpy()
        pred_unresized = np.einsum("ij,lj->lj", A_inv, (pred_lm_np - t_np[None, :]))

    return pred_landmark.cpu(), pred_unresized[...,::-1], dim_orig.cpu(), pixel_spacing.cpu(), padding.cpu()


def predict_landmarks(model, loader, device):
    """
    Predict landmarks from a loader by using the single image prediction function.
    
    Parameters:
        model (torch.nn.Module): Trained model for landmark prediction.
        loader (iterable): Data loader that yields batches of samples (each sample is a dict
                           with keys "image", "landmark", "affine", "dim_original", "spacing", "padding").
        device (torch.device): Device to run the model on.
    
    Returns:
        tuple: (pred_landmarks, pred_landmarks_transformed, dim_origs, pixel_spacings, paddings)
               where:
                 - pred_landmarks is a torch.Tensor of concatenated raw predictions,
                 - pred_landmarks_transformed is a numpy.ndarray of concatenated transformed landmarks (with channels reversed),
                 - the remaining outputs are tensors with the corresponding metadata.
    """


    all_pred_landmarks = []
    all_pred_landmarks_transformed = []
    all_dim_origs = []
    all_pixel_spacings = []
    all_paddings = []

    # Loop over the loader. Each batch is assumed to contain a batch dimension.
    for batch in tqdm(loader):
        batch_size = batch["image"].shape[0]
        for i in range(batch_size):
            # Construct a single-sample dictionary for each item in the batch.
            sample = {key: batch[key][i] for key in batch}
            pred_landmark, pred_unresized, dim_orig, pixel_spacing, padding = predict_single_image(model, sample, device)
            all_pred_landmarks.append(pred_landmark)
            all_pred_landmarks_transformed.append(pred_unresized)
            all_dim_origs.append(dim_orig)
            all_pixel_spacings.append(pixel_spacing)
            all_paddings.append(padding)

    # Concatenate results
    all_pred_landmarks = torch.cat(all_pred_landmarks, dim=0)
    # Here we assume that the transformed predictions have shape (n_landmarks, 2) for each image.
    # The final shape becomes (num_images, n_landmarks, 2) and we reverse the last axis.
    all_pred_landmarks_transformed = np.stack(all_pred_landmarks_transformed, axis=0)
    all_dim_origs = torch.stack(all_dim_origs, dim=0)
    all_pixel_spacings = torch.stack(all_pixel_spacings, dim=0)
    all_paddings = torch.stack(all_paddings, dim=0)

    return all_pred_landmarks, all_pred_landmarks_transformed, all_dim_origs, all_pixel_spacings, all_paddings
