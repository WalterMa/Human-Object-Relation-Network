# pylint: disable=wildcard-import, unused-wildcard-import
"""Model store includes pretrained model information
"""

from .resnetv1b import *
from .faster_rcnn import *
from .ho_relation import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'resnet50_v1b': resnet50_v1b,
    'resnet50_v1d': resnet50_v1d,
    'resnet101_v1d': resnet101_v1d,

    'faster_rcnn_resnet50_v1b_voc': faster_rcnn_resnet50_v1b_voc,
    'faster_rcnn_resnet50_v1b_coco': faster_rcnn_resnet50_v1b_coco,
    'faster_rcnn_resnet50_v1b_voca': faster_rcnn_resnet50_v1b_voca,
    'faster_rcnn_resnet50_v1b_custom': faster_rcnn_resnet50_v1b_custom,

    'horelation_resnet50_v1d_voca': horelation_resnet50_v1d_voca,
    'horelation_resnet50_v1d_st40': horelation_resnet50_v1d_st40,
    'horelation_resnet50_v1d_hico': horelation_resnet50_v1d_hico,
    'horelation_resnet101_v1d_st40': horelation_resnet101_v1d_st40,
    'horelation_resnet101_v1d_hico': horelation_resnet101_v1d_hico,
    }


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError('Model: %s not in:\n\t%s' % (name, '\n\t'.join(sorted(_models.keys()))))
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
