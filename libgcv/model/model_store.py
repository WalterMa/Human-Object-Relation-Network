"""Model store which provides pretrained models."""
from __future__ import print_function
import os
from mxnet.gluon.utils import check_sha1

__all__ = ['get_model_file']

_model_sha1 = {name: checksum for checksum, name in [
    ('e660d4569ccb679ec68f1fd3cce07a387252a90a', 'vgg16'),
    ('e263a9860be0a373003d011564f10701d4954fb8', 'resnet50_v1b'),
    ('117a384ecf61490eb31ea147eb0e61e6d2b8a449', 'resnet50_v1d'),
    ('1b2b825feff86b0354642a4ab59f9b6e35e47338', 'resnet101_v1d'),
    ('447328d89d70ae1e2ca49226b8d834e5a5456df3', 'faster_rcnn_resnet50_v1b_voc'),
    ('dd05f30edbb00646764fc66d994128dcac2c3e32', 'faster_rcnn_resnet50_v1b_coco'),
    ('a465eca35e78aba6ebdf99bf52031a447e501063', 'faster_rcnn_resnet101_v1d_coco'),
    ]}


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            raise AssertionError('Mismatch in the content of model file detected. Please download it again.')
    else:
        raise AssertionError('Model file: %s is not found. Please download before use it.' % file_path)
