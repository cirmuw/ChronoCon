"""
Peprocesssing transforms usefull for a dataloader
"""

import torch
from typing import Callable, Dict, Union, List, Tuple
import tqdm
from monai.transforms import (
    Transform, 
    MapTransform,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    Resized,
    SaveImaged,
    ResizeWithPadOrCropd,
    SpatialCropd, 
    SpatialCrop,
    CropForeground,
    BorderPadd
)


import pandas as pd
import numpy as np



def non_zero_indices(multi_mask: torch.Tensor,
                     non_zero: Callable[[torch.Tensor], torch.Tensor] = lambda x: x != 0
                     ) -> Dict[str, Union[torch.Size, int]]:
    """
    Obtain indices of cube with nonzero elements of the provided 3d mask.

    Args:
    -----
    multi_mask (torch.tensor): Integer mask with entries {0 = background, 1 = 'class 1', ..., num_classes].
    non_zero (Callable[[torch.tensor], torch.tensor], optional): 
        Function defining the nonzero elements in the multi_mask. 
        E.g., `non_zero = lambda x: x==1` to only consider class 1. 
        Defaults to `lambda x: x != 0`.

    Returns:
    --------
    Dict[str, Union[torch.Size, int]]: A dictionary containing:
        - shape (torch.Size): Original shape.
        - x_index_min (int): Minimum index along the x-axis.
        - x_index_max (int): Maximum index along the x-axis.
        - y_index_min (int): Minimum index along the y-axis.
        - y_index_max (int): Maximum index along the y-axis.
        - z_index_min (int): Minimum index along the z-axis.
        - z_index_max (int): Maximum index along the z-axis.

    Example:
    --------
    >>> import torch
    >>> from ra_utils.features.transforms import non_zero_indices
    >>> mm_test = torch.zeros((3,3,3), dtype=torch.int16)
    >>> mm_test[1,1,1] = 1
    >>> mm_test[1,1,2] = 2
    >>> non_zero_indices(mm_test)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 2}
    >>> non_zero_indices(mm_test, non_zero=lambda x: x==1)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 1}
    """
    shape = multi_mask.shape
    assert len(shape) == 3, "A 3d mask is expected"

    # sum nonzero elements along all but one axis
    px = (non_zero(multi_mask)).sum(axis=(1, 2))
    py = (non_zero(multi_mask)).sum(axis=(0, 2))
    pz = (non_zero(multi_mask)).sum(axis=(0, 1))

    # get corresponding indices
    i_px = torch.arange(px.shape[0])[px > 0]
    i_py = torch.arange(py.shape[0])[py > 0]
    i_pz = torch.arange(pz.shape[0])[pz > 0]

    # get min and max indices for cutting
    i_px_min = i_px.min().item()
    i_px_max = i_px.max().item()

    i_py_min = i_py.min().item()
    i_py_max = i_py.max().item()

    i_pz_min = i_pz.min().item()
    i_pz_max = i_pz.max().item()

    return {
        "shape": shape,
        "x_index_min": i_px_min,
        "x_index_max": i_px_max,
        "y_index_min": i_py_min,
        "y_index_max": i_py_max,
        "z_index_min": i_pz_min,
        "z_index_max": i_pz_max
    }


def pad_index_cube(d: dict, dx=0, dy=0, dz=0):
    """
    enlarge index cube (e.g. of nonzero elements) by dx, dy, dz in both directions
    if it does not violate the size of the original image saved in d['size'].
    Input `d` is usually the result of `non_zero_indices`. 

    Returns:
    --------
    dict with the same keys as the input. 
     'x_index_min', 'x_index_max', ... is overwritten with the new padded indices
     
    Example:
    --------
    >>> import torch
    >>> from ra_utils.features.transforms import pad_index_cube
    >>> d = {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 2}
    >>> pad_index_cube(d, dx=0, dy=0, dz=0)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 2}
    >>> pad_index_cube(d, dx=1, dy=0, dz=0)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 0, 'x_index_max': 2, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 2}
    >>> pad_index_cube(d, dx=0, dy=0, dz=1)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 0, 'z_index_max': 2}
    >>> pad_index_cube(d, dx=0, dy=100, dz=0)
    {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 0, 'y_index_max': 2, 'z_index_min': 1, 'z_index_max': 2}
    """
    for key in ['shape', 'x_index_min', 'x_index_max', 'y_index_min', 'y_index_max', 'z_index_min', 'z_index_max']: 
        assert key in d.keys()

    x_index_min_padded = max(d["x_index_min"] - dx, 0)
    y_index_min_padded = max(d["y_index_min"] - dy, 0)
    z_index_min_padded = max(d["z_index_min"] - dz, 0)

    x_index_max_padded = min(d["x_index_max"] + dx, d["shape"][0]-1)
    y_index_max_padded = min(d["y_index_max"] + dy, d["shape"][1]-1)
    z_index_max_padded = min(d["z_index_max"] + dz, d["shape"][2]-1)

    dd = d.copy()
    dd["x_index_min"] = x_index_min_padded
    dd["y_index_min"] = y_index_min_padded
    dd["z_index_min"] = z_index_min_padded

    dd["x_index_max"] = x_index_max_padded
    dd["y_index_max"] = y_index_max_padded
    dd["z_index_max"] = z_index_max_padded
    return dd


def roi_center_of_index_cube(d: dict) -> tuple:
    """

     
    Example:
    --------
    >>> import torch
    >>> from ra_utils.features.transforms import roi_center_of_index_cube
    >>> d = {'shape': torch.Size([3, 3, 3]), 'x_index_min': 1, 'x_index_max': 1, 'y_index_min': 1, 'y_index_max': 1, 'z_index_min': 1, 'z_index_max': 2}
    >>> roi_center_of_index_cube(d)
    [1, 1, 1]
    >>> d = {'shape': torch.Size([3, 3, 3]), 'x_index_min': 0, 'x_index_max': 2, 'y_index_min': 2, 'y_index_max': 2, 'z_index_min': 1, 'z_index_max': 2}
    >>> roi_center_of_index_cube(d)
    [1, 2, 1]
    """
    for key in ['shape', 'x_index_min', 'x_index_max', 'y_index_min', 'y_index_max', 'z_index_min', 'z_index_max']: 
        assert key in d.keys()

    # round down
    return [(d["x_index_max"] + d["x_index_min"]) // 2, 
            (d["y_index_max"] + d["y_index_min"]) // 2, 
            (d["z_index_max"] + d["z_index_min"]) // 2]



def nonzero_index_cube_dataframe(loader, ids: List[str], 
                                 dx: int=0, dy: int=0, dz: int=0) -> pd.DataFrame:
    " Get indexes of nonzero (cube) of masks for all patients as dataframe"

    rows = []
    for patient_id in tqdm(ids):
        mm = loader[patient_id]
        d = non_zero_indices(mm, non_zero= lambda x: x != 0)
        d = pad_index_cube(d, dx=dx, dy=dy, dz=dz)
        index_dict = {"id": patient_id} | d
        rows.append(index_dict)
    no_zero_indices_df = pd.DataFrame(rows)
    return no_zero_indices_df


def add_index_differences(df: pd.DataFrame) -> None:
    dix = df["x_index_max"] - df["x_index_min"]
    diy = df["y_index_max"] - df["y_index_min"]
    diz = df["z_index_max"] - df["z_index_min"]
    df["dix"] = dix
    df["diy"] = diy
    df["diz"] = diz
    return None




class CropToROId(MapTransform):
    def __init__(self, keys, 
                 non_zero: Callable[[torch.Tensor], torch.Tensor] = lambda x: x != 0,
                 crop_x=False, 
                 crop_y=False,
                 crop_z=True, 
                 dx=0, 
                 dy=0, 
                 dz=0):
        super().__init__(keys)
        self.non_zero = non_zero
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z

        self.dx=dx
        self.dy=dy
        self.dz=dz

    def __call__(self, data):
        d = dict(data)
        roi = non_zero_indices(d['label'][0, ...], self.non_zero)
        roi = pad_index_cube(roi, dx=self.dx, dy=self.dy, dz=self.dz)
        x_min, x_max = roi['x_index_min'], roi['x_index_max'] + 1
        y_min, y_max = roi['y_index_min'], roi['y_index_max'] + 1
        z_min, z_max = roi['z_index_min'], roi['z_index_max'] + 1
        for key in self.keys:
            if self.crop_x:
                d[key] = d[key][:, x_min:x_max, :, :]
            if self.crop_y:
                d[key] = d[key][:, :, y_min:y_max, :]
            if self.crop_z:
                d[key] = d[key][:, :, :, z_min:z_max]
        return d


class SpatialCropToROId(MapTransform):
    """
    Crop a region of interest (ROI) from the input data based on non-zero values in the label map.

    This transform identifies the bounding box of non-zero values in the label map, 
    expands it by a specified margin, and crops the input data to the defined ROI.

    Parameters
    ----------
    keys : list of str
        Keys of the corresponding items to be transformed (e.g. ["image", "label"] )
    roi_size : tuple of int
        (x,y,z) size of output
    non_zero : callable, optional
        A function to identify non-zero values in the label map. Default is a lambda function `lambda x: x != 0`.
    pedantic : bool, optional
        If True, raises an exception if the ROI does not fit within the given roi_size. If False, prints a warning.
        Default is True.
    dx : int, optional
        Margin to add to the ROI in the x-dimension. Default is 0.
        Only used for checking if it fits
    dy : int, optional
    dz : int, optional
    verbose: bool, optional
        If True, prints additional information during processing. Default is True.

    Methods
    -------
    __call__(data)
        Apply the transformation to the input data.

    Examples
    --------
    > transform = SpatialCropToROId(keys=["image", "label"], roi_size=(128, 128, 64))
    > transformed = transform({"image": image_tensor, "label": label_tensor})
    """
    def __init__(self, keys, roi_size: Tuple[int],
                 non_zero: Callable[[torch.Tensor], torch.Tensor] = lambda x: x != 0, 
                 pedantic = True, 
                 dx=0, # only used to check if roi fits into output_roi size
                 dy=0, 
                 dz=0, 
                 use_center_instead = [False, False, False],
                 verbose=True):
        super().__init__(keys)
        self.non_zero = non_zero
        self.roi_size = roi_size
        self.pedantic = pedantic

        self.dx=dx
        self.dy=dy
        self.dz=dz
        self.use_center_instead=use_center_instead
        self.verbose = verbose

        #self.cropper = CropForeground(select_fn=non_zero, margin=margin)


    def __call__(self, data):
        d = dict(data)
        roi = non_zero_indices(d['label'][0, ...], self.non_zero)
        roi = pad_index_cube(roi, dx=self.dx, dy=self.dy, dz=self.dz)

        # check if fits 
        roi_max = (roi["x_index_max"], roi["y_index_max"], roi["z_index_max"])
        roi_min = (roi["x_index_min"], roi["y_index_min"], roi["z_index_min"])

        if self.verbose:
            print("needed size: ", np.array(roi_max) - np.array(roi_min) + np.array([1,1,1]))
        for i in range(3):
            roi_max_ = roi_max[i]; 
            roi_min_ = roi_min[i]; 
            d_roi = roi_max_ - roi_min_ + 1
            if d_roi > self.roi_size[i]:
                if self.pedantic:
                    raise Exception(f"ROI does not fit given roi_size: dim = {i} needed size = {d_roi} > {self.roi_size[i]} = given size")
                else:
                    print(f"WARINING: ROI does not fit given roi_size: dim = {i} needed size = {d_roi} > {self.roi_size[i]} = given size")
                
                
        roi_center = roi_center_of_index_cube(roi)
        for i, use_center in enumerate(self.use_center_instead):
            if use_center:
                roi_center[i] = d['label'].shape[i+1] // 2

        cropper = SpatialCrop(roi_center=roi_center, 
                              roi_size=self.roi_size)
        for key in self.keys:
            d[key] = cropper(d[key])
        return d
