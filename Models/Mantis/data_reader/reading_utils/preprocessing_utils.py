import numpy as np


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data, max_length):
    """
    TODO: pad_length cannot be negative? maybe better to call enlarge_dim_by_padding
    """
    # via this it can works on more dimensional array
    pad_length = max_length - data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape) * np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
