from pathlib import Path
import pandas as pd

import ra_utils

import landmarker
from landmarker.data import LandmarkDataset, LandmarkDatasetOnTheFly
from typing import Union, List

def get_landmark_datasets_v2(
    image_paths_train,
    image_paths_test1,
    image_paths_test2,
    landmarks_train,
    landmarks_test1,
    landmarks_test2,
    train_transform=None,
    inference_transform=None,
    dim_img=(512,512),    
    pixel_spacings_train=None,
    pixel_spacings_test1=None,
    pixel_spacings_test2=None,
    class_names: Union[List[str], None] = None,
    **kwargs
    ):
    return (
        LandmarkDatasetOnTheFly(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_train) > 0 else None,
        LandmarkDatasetOnTheFly(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test1) > 0 else None,
        LandmarkDatasetOnTheFly(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test2) > 0 else None,
    )


def get_landmark_datasets(
    image_paths_train,
    image_paths_test1,
    image_paths_test2,
    landmarks_train,
    landmarks_test1,
    landmarks_test2,
    train_transform=None,
    inference_transform=None,
    dim_img=(512,512),    
    pixel_spacings_train=None,
    pixel_spacings_test1=None,
    pixel_spacings_test2=None,
    class_names: Union[List[str], None] = None,
    **kwargs
    ):
    return (
        LandmarkDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_train) > 0 else None,
        LandmarkDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test1) > 0 else None,
        LandmarkDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test2) > 0 else None,
    )



def get_landmark_datasets(
    image_paths_train,
    image_paths_test1,
    image_paths_test2,
    landmarks_train,
    landmarks_test1,
    landmarks_test2,
    train_transform=None,
    inference_transform=None,
    dim_img=(512,512),    
    pixel_spacings_train=None,
    pixel_spacings_test1=None,
    pixel_spacings_test2=None,
    class_names: Union[List[str], None] = None,
    **kwargs
    ):
    return (
        LandmarkDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_train) > 0 else None,
        LandmarkDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test1) > 0 else None,
        LandmarkDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            dim_img=dim_img,
            class_names=class_names,
            **kwargs,
        ) if len(image_paths_test2) > 0 else None,
    )



# Example ::
#
# folds_data_F = dataHandler.get_landmarks_dataset_F_CV()
# cv_datasets_feet = get_landmark_datasets_CV(
#     folds_data=folds_data_F,
#     train_transform=my_train_transform,
#     inference_transform=my_test_transform,
#     dim_img=(512, 512),
#     pixel_spacings_folds=pixel_spacings_folds_F
# )

def get_landmark_datasets_CV(
    folds_data,
    train_transform=None,
    inference_transform=None,
    dim_img=(512, 512),
    pixel_spacings_folds=None,
    **kwargs
):
    """
    Parameters
    ----------
    folds_data : list
        Typically the output of a function like get_landmarks_dataset_H_CV() or get_landmarks_dataset_F_CV().
        Each element should be a tuple of:
            (train_image_paths, train_landmarks, test_image_paths, test_landmarks)
    train_transform : callable, optional
        Transform to apply to the training dataset.
    inference_transform : callable, optional
        Transform to apply to the test dataset.
    dim_img : tuple, optional
        (height, width) to resize or pad images in the test sets, etc.
    pixel_spacings_folds : list, optional
        A parallel list to folds_data, containing pixel spacings for each fold.
        If provided, each element should map to something like:
            pixel_spacings_folds[i]['train'] -> list of pixel spacings for train images
            pixel_spacings_folds[i]['test']  -> list of pixel spacings for test images
        or None if spacing is not being used.
    **kwargs : dict
        Additional arguments passed to the LandmarkDataset constructor.

    Returns
    -------
    list
        A list of length N (number of folds). Each element is a tuple:
            (train_dataset, test_dataset)
        Each is a LandmarkDataset instance.
    """

    fold_datasets = []

    # If pixel_spacings_folds is not provided, default to None for each fold
    use_spacings = (pixel_spacings_folds is not None)
    
    for i, fold_info in enumerate(folds_data):
        if len(fold_info) != 4:
            raise ValueError(
                "Each element in folds_data must be a tuple of "
                "(train_image_paths, train_landmarks, test_image_paths, test_landmarks)."
            )
        
        (train_image_paths, train_landmarks,
         test_image_paths, test_landmarks) = fold_info

        # Pixel spacings (optional)
        if use_spacings:
            train_pixel_spacings = pixel_spacings_folds[i].get("train", None)
            test_pixel_spacings  = pixel_spacings_folds[i].get("test", None)
        else:
            train_pixel_spacings = None
            test_pixel_spacings  = None

        # Create the training dataset
        train_dataset = LandmarkDataset(
            train_image_paths,
            train_landmarks,
            pixel_spacing=train_pixel_spacings,
            transform=train_transform,
            dim_img=dim_img,
            **kwargs
        )

        # Create the test dataset
        test_dataset = LandmarkDataset(
            test_image_paths,
            test_landmarks,
            pixel_spacing=test_pixel_spacings,
            transform=inference_transform,
            dim_img=dim_img,
            **kwargs
        )

        fold_datasets.append((train_dataset, test_dataset))

    return fold_datasets
