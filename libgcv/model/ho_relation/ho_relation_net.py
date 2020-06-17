"""Human-object Relation Network."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet.gluon import nn
from .base import HORelationBase
from libgcv.model.ho_relation.module import HumanObjectRelationModule


class HORelationNet(HORelationBase):
    r"""Human-object Relation Network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training.
    train_post_nms : int, default is 2000
        Return top proposal results after NMS in training.
    test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing.
    test_post_nms : int, default is 300
        Return top proposal results after NMS in testing.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    test_gt_box_input: boolean, default is False
        When true, require and use gt_box as proposal in test forward.

    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.

    """
    def __init__(self, features, top_features, classes,
                 short=600, max_size=1000, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=10, pos_iou_thresh=0.5, num_ctx_per_sample=10,
                 ctx_iou_lb=0.2, ctx_iou_ub=0.75, additional_output=False, global_avg_pool=True,
                 **kwargs):
        super(HORelationNet, self).__init__(
            features=features, top_features=top_features, classes=classes,
            short=short, max_size=max_size, train_patterns=train_patterns,
            nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms,
            roi_mode=roi_mode, roi_size=roi_size, stride=stride, clip=clip,
            global_avg_pool=global_avg_pool, **kwargs)
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._num_ctx_per_sample = num_ctx_per_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        # Use {} to warp non HybridBlock
        self._additional_output = additional_output

        with self.name_scope():
            self.fc = nn.Dense(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.fc_ctx = nn.Dense(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.relation = HumanObjectRelationModule(num_feat=1024, num_group=16, additional_output=additional_output)
            self.class_predictor = nn.Dense(
                self.num_class, weight_initializer=mx.init.Normal(0.01))
            self.ctx_class_predictor = nn.Dense(
                self.num_class, weight_initializer=mx.init.Normal(0.01))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None, obj_box=None):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : mxnet.nd.NDArray or mxnet.symbol
            The ground-truth bbox tensor with shape (1, N, 4).
        obj_box : mxnet.nd.NDArray or mxnet.symbol
            The object bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """
        feat = self.features(x)
        rsn_box = obj_box.reshape((-1, 4))

        # create batchid
        rsn_batchid = F.zeros_like(rsn_box.slice_axis(axis=-1, begin=0, end=1))
        rsn_rois = F.concat(*[rsn_batchid, rsn_box], dim=-1)
        gt_batchid = F.zeros_like(gt_box.slice_axis(axis=-1, begin=0, end=1))
        gt_rois = F.concat(*[gt_batchid.reshape((-1, 1)), gt_box.reshape((-1, 4))], dim=-1)

        # ROI features
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, gt_rois, self._roi_size, 1. / self._stride)
            pooled_ctx_feat = F.ROIPooling(feat, rsn_rois, self._roi_size, 1. / self._stride)
        elif self._roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, gt_rois, self._roi_size, 1. / self._stride,
                                             sample_ratio=2)
            pooled_ctx_feat = F.contrib.ROIAlign(feat, rsn_rois, self._roi_size, 1. / self._stride,
                                                 sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        top_feat = self.top_features(pooled_feat)
        # contextual region prediction
        top_ctx_feat = self.top_features(pooled_ctx_feat)

        if self.use_global_avg_pool:
            top_feat = self.global_avg_pool(top_feat)
            top_ctx_feat = self.global_avg_pool(top_ctx_feat)

        top_feat = self.fc(top_feat)
        top_ctx_feat = self.fc_ctx(top_ctx_feat)
        if self._additional_output:
            relation_feat, relation_ctx_feat, relation = \
                self.relation(top_feat, top_ctx_feat, gt_box.reshape((-1, 4)), rsn_box)
        else:
            relation_feat, relation_ctx_feat = \
                self.relation(top_feat, top_ctx_feat, gt_box.reshape((-1, 4)), rsn_box)
        top_feat = top_feat + relation_feat
        top_ctx_feat = top_ctx_feat + relation_ctx_feat

        cls_pred = self.class_predictor(top_feat)
        ctx_cls_pred = self.ctx_class_predictor(top_ctx_feat)

        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, -1, self.num_class))
        ctx_cls_pred = ctx_cls_pred.reshape((self._max_batch, -1, self.num_class))

        ctx_cls_pred = ctx_cls_pred.max(axis=1, keepdims=True)
        cls_pred = F.broadcast_add(cls_pred, ctx_cls_pred)

        if self._additional_output:
            return cls_pred, relation
        return cls_pred


def get_horelation(name, dataset, pretrained=False, params='', ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return a network.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = HORelationNet(**kwargs)
    if pretrained:
        if params.strip():
            net.load_parameters(params.strip())
        else:
            from ..model_store import get_model_file
            full_name = '_'.join(('horelation', name, dataset))
            net.load_parameters(get_model_file(full_name, root=root), ctx=ctx)
    return net


def horelation_resnet50_v1d_voca(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        from ..resnetv1b import resnet50_v1d
        from ...data import VOCAction
        classes = VOCAction.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet50_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet50_v1d', dataset='voca', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=50, rpn_train_post_nms=50,
            rpn_test_pre_nms=50, rpn_test_post_nms=50, rpn_min_size=16,
            num_sample=10, pos_iou_thresh=0.5, num_ctx_per_sample=10,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75,  params=params,
            **kwargs)
    else:
        raise NotImplementedError


def horelation_resnet50_v1d_st40(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        from ..resnetv1b import resnet50_v1d
        from ...data import Stanford40Action
        classes = Stanford40Action.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet50_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet50_v1d', dataset='st40', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=300, rpn_train_post_nms=100,
            rpn_test_pre_nms=300, rpn_test_post_nms=100, rpn_min_size=16,
            num_sample=5, pos_iou_thresh=0.5, num_ctx_per_sample=20,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75, params=params,
            **kwargs)
    else:
        raise NotImplementedError


def horelation_resnet50_v1d_hico(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        from ..resnetv1b import resnet50_v1d
        from ...data import HICOClassification
        classes = HICOClassification.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet50_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet50_v1d', dataset='hico', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=300, rpn_train_post_nms=100,
            rpn_test_pre_nms=300, rpn_test_post_nms=100, rpn_min_size=16,
            num_sample=10, pos_iou_thresh=0.5, num_ctx_per_sample=10,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75,  params=params,
            **kwargs)
    else:
        raise NotImplementedError


def horelation_resnet101_v1d_st40(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        from ..resnetv1b import resnet101_v1d
        from ...data import Stanford40Action
        classes = Stanford40Action.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet101_v1d', dataset='st40', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=6000, rpn_train_post_nms=100,
            rpn_test_pre_nms=6000, rpn_test_post_nms=100, rpn_min_size=16,
            num_sample=5, pos_iou_thresh=0.5, num_ctx_per_sample=20,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75, params=params,
            **kwargs)
    else:
        raise NotImplementedError


def horelation_resnet101_v1d_hico(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        from ..resnetv1b import resnet101_v1d
        from ...data import HICOClassification
        classes = HICOClassification.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet101_v1d', dataset='hico', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=6000, rpn_train_post_nms=100,
            rpn_test_pre_nms=6000, rpn_test_post_nms=100, rpn_min_size=16,
            num_sample=5, pos_iou_thresh=0.5, num_ctx_per_sample=20,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75, params=params,
            **kwargs)
    else:
        raise NotImplementedError
