from __future__ import division
from __future__ import print_function

import sys
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# add module path
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
import argparse
import glob
import logging
logging.basicConfig(level=logging.INFO)
import mxnet as mx
from tqdm import tqdm
import libgcv
from libgcv import data as gdata
from libgcv.data import batchify
from libgcv.data.transforms.presets.horelation import HORelationDefaultValTransform
from libgcv.utils.metrics.voc_multi_classification import VOCMultiClsMApMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Validate Human-object Relation networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1d',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='hico',
                        help='Testing dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--eval-all', action='store_true',
                        help='Eval all models begins with save prefix. Use with pretrained.')
    parser.add_argument('--save-outputs', action='store_true',
                        help='Save model outputs include labels.')
    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'hico':
        val_dataset = gdata.HICOClassification(split='test', preload_label=args.eval_all, load_box=True, ignore_label=-1)
        val_metric = VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, hico_ap_type=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(HORelationDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return val_loader


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, size):
    """Test on validation dataset."""
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    with tqdm(total=size) as pbar:
        for batch in val_data:
            batch = split_and_load(batch, ctx_list=ctx)
            cls_scores = []
            gt_classes = []
            for data, label, box in zip(*batch):
                gt_box = label[:, :, :4]
                # get prediction results
                cls_score = net(data, gt_box, box)
                # MIL for image level classification, shape (B, C)
                cls_score = mx.nd.sigmoid(cls_score.max(axis=1, keepdims=False))
                cls_scores.append(cls_score[:, :])
                gt_classes.append(label[:, 0, 5:])

            # update metric
            for score, gt_class in zip(cls_scores, gt_classes):
                eval_metric.update(score, gt_class)
            pbar.update(len(ctx))
    return eval_metric.get()


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('horelation', args.network, args.dataset))
    args.save_prefix += net_name
    net = libgcv.model.get_model(net_name, pretrained=False)

    # testing data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, args.batch_size, args.num_workers)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_eval.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)

    # validation
    if not args.eval_all:
        logger.info('[Model: {}] Start Evaluation'.format(args.pretrained))
        net.load_parameters(args.pretrained.strip())
        net.collect_params().reset_ctx(ctx)
        map_name, mean_ap = validate(net, val_data, ctx, eval_metric, len(val_dataset))
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Model: {}] Evaluation Result: \n{}'.format(args.pretrained, val_msg))
        if args.save_outputs:
            saved_name = net_name + '_eval_outputs.csv'
            eval_metric.save(file_name=saved_name)
    else:
        saved_models = glob.glob(args.save_prefix + '*.params')
        for index, saved_model in enumerate(sorted(saved_models)):
            logger.info('[Index {}] Start Evaluating from {}'.format(index, saved_model))
            net.load_parameters(saved_model)
            net.collect_params().reset_ctx(ctx)
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, len(val_dataset))
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Index {}] Evaluation Result: \n{}'.format(index, val_msg))
            logger.info('[Index {}] Complete Evaluating from {}'.format(index, saved_model))
            current_map = float(mean_ap[-1])
            with open(args.save_prefix+'_best_eval_map.log', 'a') as f:
                f.write('\nModel {}:\t{:.4f}'.format(saved_model, current_map))
            if args.save_outputs:
                saved_name = net_name + '_index_{:04d}_eval_outputs.csv'.format(index)
                eval_metric.save(file_name=saved_name)
