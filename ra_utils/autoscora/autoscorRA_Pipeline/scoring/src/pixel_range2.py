import numpy as np
#import cv2


# pixel range correction
def pixel_convert(img, target_range=(0, 1),
                  clip_bottom=None, clip_top=None,
                  target_dtype=np.float32):
    if clip_top is None:
        clip_top = 100
    if len(img.shape) == 3:
        clipped_img = np.clip(img,
                              a_min=np.percentile(img, clip_bottom),
                              a_max=np.percentile(img, clip_top))
        range_adj_img = (clipped_img - np.min(clipped_img))/(np.max(clipped_img) - np.min(clipped_img))
        range_adj_img = range_adj_img * (target_range[1] - target_range[0]) + target_range[0]
        dtype_adj_img = range_adj_img.astype(dtype=target_dtype)
        final_clip_img = np.clip(dtype_adj_img, a_min=target_range[0], a_max=target_range[1])
    else:
        raise ValueError('img.shape is not equal to 3.')
    return final_clip_img


def reshape_with_padding(array, target_shape):

    """
    Reshapes a numpy array to target_shape. If aspect ratio of array differs from that of target_shape, approriate
    padding is applied to x or y axis. Only then resampling is done, so that the input array is not distorted.

    :param array: 2D numpy array
    :param target_shape: int, (height, width)
    :return: padded and reshaped numpy array
    """

    target_ratio = target_shape[0]/target_shape[1]
    array_shape = array.shape
    array_ratio = array_shape[0] / array_shape[1]

    if array_ratio > target_ratio:
        padded_width = round(array_ratio / target_ratio * array_shape[1], 5)
        # "round" above is used to discard some decimals to account for float precision errors
        # otherwise np.floor or np.ceil would subtract/add one int when padded_height is actually exactly an int
        pad_horiz_left = int(np.floor((padded_width - array_shape[1]) / 2))
        pad_horiz_right = int(np.ceil((padded_width - array_shape[1]) / 2))
        padded_array = np.pad(array, ((0, 0), (pad_horiz_left, pad_horiz_right)), 'constant', constant_values=0)
    elif array_ratio < target_ratio:
        padded_height = round(target_ratio / array_ratio * array_shape[0], 5)
        # "round" above is used to discard some decimals to account for float precision errors
        # otherwise np.floor or np.ceil would subtract/add one int when padded_height is actually exactly an int
        pad_vert_up = int(np.floor((padded_height - array_shape[0]) / 2))
        pad_vert_down = int(np.ceil((padded_height - array_shape[0]) / 2))
        padded_array = np.pad(array, ((pad_vert_up, pad_vert_down), (0, 0)), 'constant', constant_values=0)
    else:
        padded_array = array

    assert abs(padded_array.shape[0]/padded_array.shape[1] - target_ratio) <= 0.005

    reshaped_padded_array = cv2.resize(padded_array, (target_shape[1], target_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
    return reshaped_padded_array
