from __future__ import division
from __future__ import print_function

import numpy as np


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding size.

    Args:
        pad: padding, "SAME", "VALID" or manually specified tuple [P, Q].
        ksize: kernel size, [I, J].

    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        return (out_siz - 1) * stride + ksize - in_siz
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.

    Args:
        h: input image size.
        kh: kernel size.
        pad: padding strategy.
        sh: stride.

    Returns:
        s: output size.
    """

    if pad == 'VALID':
        return np.ceil((h - kh + 1) / sh)
    elif pad == 'SAME':
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))


def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Convert tensor to sliding windows.

    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    if type(pad) == str:
        h2 = int(calc_size(h, kh, pad, sh))
        w2 = int(calc_size(w, kw, pad, sw))
        ph = int(calc_pad(pad, h, h2, sh, kh))
        pw = int(calc_pad(pad, w, w2, sw, kw))
    else:
        h2 = int(calc_size(h, kh, pad[0], sh))
        w2 = int(calc_size(w, kw, pad[1], sw))
        ph = int(calc_pad(pad[0], h, h2, sh, kh))
        pw = int(calc_pad(pad[1], w, w2, sw, kw))

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2))
    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)
    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)
    x = np.pad(x, ((0, 0), pph, ppw, (0, 0)),
               mode='constant', constant_values=(0.0,))
    y = np.zeros([n, h2, w2, kh, kw, c])
    for ii in range(h2):
        for jj in range(w2):
            xx = ii * sh
            yy = jj * sw
            y[:, ii, jj, :, :, :] = x[:, xx: xx + kh, yy: yy + kw, :]
    return y


def conv2d(x, w, pad='SAME'):
    """2D stride 1 convolution (technically speaking, correlation).

    Args:
        x: [N, H, W, C]
        w: [I, J, C, K]
        pad: 'SAME', 'VALID', or a tuple [PH, PW]

    Returns:
        y: [N, H', W', K]
    """
    ksize = w.shape[: 2]
    x = extract_sliding_windows(x, ksize, pad=pad, stride=(1, 1))
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = x.dot(w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y
