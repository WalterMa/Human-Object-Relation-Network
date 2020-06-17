from __future__ import division
from __future__ import print_function

import sys
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# add module path
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import mxnet as mx
from tqdm import tqdm
from libgcv import data as gdata
import gluoncv
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Faster-RCNN networks.')
    parser.add_argument('--dataset', type=str, default='hico',
                        help='Testing dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voca':
        val_dataset = gdata.VOCAction(split='test')
    elif dataset.lower() == 'st40':
        val_dataset = gdata.Stanford40Action(split='test')
    elif dataset.lower() == 'hico':
        val_dataset = gdata.HICOClassification(split='all', preload_label=False)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset

def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return val_loader

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def validate(net, val_data, ctx, dataset):
    """Test on validation dataset."""
    clipper = gluoncv.nn.bbox.BBoxClipToImage()
    net.hybridize(static_alloc=True)
    with tqdm(total=len(dataset)) as pbar:
        idx = 0
        for ib, batch in enumerate(val_data):
            batch = split_and_load(batch, ctx_list=ctx)
            for x, y, im_scale in zip(*batch):
                # get prediction results
                ids, scores, bboxes = net(x)
                # clip to image size
                bboxes = clipper(bboxes[:, 0:100, :], x)
                # rescale to original resolution
                im_scale = im_scale.reshape((-1)).asscalar()
                bboxes *= im_scale
                bboxes = bboxes.reshape((-1, 4)).asnumpy()
                bboxes = bboxes[bboxes.sum(axis=1).nonzero()]
                # split ground truths
                dataset.save_boxes(idx, bboxes)
                idx += 1
            pbar.update(len(ctx))


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = 'faster_rcnn_fpn_resnet101_v1d_coco'
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gluoncv.model_zoo.get_model(net_name, pretrained=True)
    else:
        net = gluoncv.model_zoo.get_model(net_name, pretrained=False)
        net.load_parameters(args.pretrained.strip())
    net.set_nms(nms_thresh=0.5, nms_topk=100, post_nms=100, force_nms=True)
    net.collect_params().reset_ctx(ctx)


    # testing data
    val_dataset = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, args.batch_size, args.num_workers)

    # validation
    validate(net, val_data, ctx, val_dataset)
