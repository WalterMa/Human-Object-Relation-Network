"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import random
import mxnet as mx
from .image import plot_image

def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituded.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')
    return ax


def plot_with_ctx_bbox(img, scores, gt_boxes, ctx_boxes, ax=None, class_names=None, reverse_rgb=False):
    """Visualize bounding boxes With contextual boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N, num_class`.
    gt_boxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    ctx_boxes : numpy.ndarray or mxnet.nd.NDArray
        Contextual bounding boxes with shape `N, num_class, 4`. Where `N` is the number of boxes.
    class_names : list of str, optional
        Description of parameter `class_names`.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if not len(gt_boxes) == len(ctx_boxes):
        raise ValueError('The length of ctx_boxes and gt_boxes mismatch, {} vs {}'
                         .format(len(ctx_boxes), len(gt_boxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(gt_boxes) < 1:
        return ax

    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(gt_boxes, mx.nd.NDArray):
        gt_boxes = gt_boxes.asnumpy()
    if isinstance(ctx_boxes, mx.nd.NDArray):
        ctx_boxes = ctx_boxes.asnumpy()

    for i in range(len(gt_boxes)):
        cls_id = scores[i].argmax()
        xmin, ymin, xmax, ymax = [int(x) for x in ctx_boxes[i, cls_id]]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor='g',
                             linewidth=3.5)
        ax.add_patch(rect)
        xmin, ymin, xmax, ymax = [int(x) for x in gt_boxes[i]]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor='r',
                             linewidth=3.5)
        ax.add_patch(rect)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores[i, cls_id]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor='r', alpha=0.5),
                    fontsize=12, color='white')
    return ax


def plot_gt_and_ctx_boxes(img, gt_boxes, ctx_boxes, ax=None, reverse_rgb=False):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    gt_boxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4,`.
    ctx_boxes : numpy.ndarray or mxnet.nd.NDArray, optional
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if isinstance(gt_boxes, mx.nd.NDArray):
        gt_boxes = gt_boxes.asnumpy().reshape((-1, 4))
    if isinstance(ctx_boxes, mx.nd.NDArray):
        ctx_boxes = ctx_boxes.asnumpy().reshape((-1, 4))

    gt_color = (46/255, 117/255, 182/255)  # Blue
    ctx_color = (255/255, 192/255, 0/255)  # Yellow

    # plot gt box
    for i, bbox in enumerate(gt_boxes):
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=gt_color,
                             linewidth=3.5)
        ax.add_patch(rect)

    for i, bbox in enumerate(ctx_boxes):
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=ctx_color,
                             linewidth=3.5)
        ax.add_patch(rect)
    return ax


def plot_single_gt_with_realtion(img, gt_box, ctx_boxes, relation_weights,
                                 thresh=0.1, ax=None, reverse_rgb=False):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    gt_box : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `4,`.
    ctx_boxes : numpy.ndarray or mxnet.nd.NDArray, optional
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    relation_weights : numpy.ndarray or mxnet.nd.NDArray
        Relation weights of the provided gt_box and ctx_boxes, shape (N,)
    thresh : float, optional, default 0.5
        Display threshold for relations.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if isinstance(gt_box, mx.nd.NDArray):
        gt_box = gt_box.asnumpy().reshape((-1,))
    if isinstance(ctx_boxes, mx.nd.NDArray):
        ctx_boxes = ctx_boxes.asnumpy().reshape((-1, 4))
    if isinstance(relation_weights, mx.nd.NDArray):
        relation_weights = relation_weights.asnumpy().reshape((-1,))

    gt_color = (46/255, 117/255, 182/255)  # Blue
    pos_color = (255/255, 192/255, 0/255)  # Yellow
    neg_color = (192/255, 0/255, 0/255)    # Red

    # plot gt box
    xmin, ymin, xmax, ymax = [int(x) for x in gt_box]
    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                         ymax - ymin, fill=False,
                         edgecolor=gt_color,
                         linewidth=3.5)
    ax.add_patch(rect)

    ctx_color = None
    for i, bbox in enumerate(ctx_boxes):
        if relation_weights[i] <= thresh:
            ctx_color = neg_color
        else:
            ctx_color = pos_color
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=ctx_color,
                             linewidth=3.5)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, '{:.3f}'.format(float(relation_weights[i])),
                bbox=dict(facecolor=ctx_color, alpha=0.5), fontsize=10, color='white')
    return ax


def plot_single_gt_with_box_and_realtion(img, gt_box, ctx_boxes, box_weights, relation_weights,
                                         thresh=0.1, ax=None, reverse_rgb=False):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    gt_box : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `4,`.
    ctx_boxes : numpy.ndarray or mxnet.nd.NDArray, optional
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    box_weights :  numpy.ndarray or mxnet.nd.NDArray
        Box weights of the ctx_boxes, shape (N,)
    relation_weights : numpy.ndarray or mxnet.nd.NDArray
        Relation weights of the provided gt_box and ctx_boxes, shape (N,)
    thresh : float, optional, default 0.5
        Display threshold for relations.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if isinstance(gt_box, mx.nd.NDArray):
        gt_box = gt_box.asnumpy().reshape((-1,))
    if isinstance(ctx_boxes, mx.nd.NDArray):
        ctx_boxes = ctx_boxes.asnumpy().reshape((-1, 4))
    if isinstance(box_weights, mx.nd.NDArray):
        relation_weights = relation_weights.asnumpy().reshape((-1,))
    if isinstance(relation_weights, mx.nd.NDArray):
        relation_weights = relation_weights.asnumpy().reshape((-1,))

    gt_color = (46/255, 117/255, 182/255)  # Blue
    box_color = (255/255, 192/255, 0/255)  # Yellow
    # relation_color = (192/255, 0/255, 0/255)    # Red

    # plot gt box
    xmin, ymin, xmax, ymax = [int(x) for x in gt_box]
    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                         ymax - ymin, fill=False,
                         edgecolor=gt_color,
                         linewidth=3.5)
    ax.add_patch(rect)

    for i, bbox in enumerate(ctx_boxes):
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=box_color,
                             linewidth=3.5)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, '{:.3f}/{:.3f}'.format(float(box_weights[i]), float(relation_weights[i])),
                bbox=dict(facecolor=box_color, alpha=0.5), fontsize=10, color='white')
    return ax

