"""Transforms for RCNN series."""
from __future__ import absolute_import
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage

__all__ = ['load_single_test']


def load_single_test(img, bboxes, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """A util function to load a single image, transform them to tensor by applying
    normalizations. This function support 1 filename.

    Parameters
    ----------
    image : numpy.ndarray
        Image numpy.ndarray, (C, H, W)
    bboxes: numpy.ndarray
        Bounding boxes coordinates of persons in the Image. Shape should be (N, 4).
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray, mxnet.NDArray)
        A (1, 3, H, W) mxnet NDArray as image input to network,
        a numpy ndarray as original un-normalized color image for display,
        a (1, N, 4) mxnet NDArray as boxes input to network.

    """
    h, w, _ = img.shape
    img = timage.resize_short_within(img, short, max_size, interp=1)
    ori_img = img.asnumpy().astype('uint8')
    bboxes = tbbox.resize(bboxes, (w, h), (img.shape[1], img.shape[0]))
    bboxes = mx.nd.array(bboxes)
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    return img.expand_dims(0), ori_img, bboxes.expand_dims(0)
