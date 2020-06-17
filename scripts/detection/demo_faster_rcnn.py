"""Faster RCNN Demo script."""
import os
import sys
# add module path
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
import argparse
import mxnet as mx
import libgcv
from libgcv.data.transforms import presets
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with Faster RCNN networks.')
    parser.add_argument('--network', type=str, default='faster_rcnn_resnet50_v1b_voc',
                        help="Faster RCNN full network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    if not args.images.strip():
        raise AssertionError("Please specify test images.")
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = libgcv.model.get_model(args.network, pretrained=True)
    else:
        net = libgcv.model.get_model(args.network, pretrained=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.3, 200)

    ax = None
    for image in image_list:
        x, img = presets.rcnn.load_test(image, short=net.short, max_size=net.max_size)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        ax = libgcv.utils.viz.plot_bbox(img, bboxes, scores, ids,
                                        class_names=net.classes, ax=ax)
        plt.show()
