import numpy as np
from skimage.metrics import structural_similarity

def compute_ssim(im1, im2):
    assert im1.shape[-1] == 1 and im2.shape[-1] == 1
    _ssim = []
    for x1, x2 in zip(im1, im2):
        x1, x2 = x1[...,0], x2[...,0]
        _ssim.append(structural_similarity(x1, x2, win_size=11, data_range=1.0))

    return np.array(_ssim)