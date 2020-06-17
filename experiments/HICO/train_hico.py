import argparse
import sys
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# add module path
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
import logging
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import libgcv
from libgcv import data as gdata
from libgcv import utils as gutils
from libgcv.model import get_model
from libgcv.data import batchify
from libgcv.data.transforms.presets.horelation import HORelationDefaultTrainTransform, HORelationDefaultValTransform
from libgcv.utils.metrics.voc_multi_classification import VOCMultiClsMApMetric
from libgcv.nn.lr_schedule import CosineAnnealingSchedule


def parse_args():
    parser = argparse.ArgumentParser(description='Train Human-object Relation networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1d',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='hico',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='15',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--max-lr', type=str, default='4e-5',
                        help='Max learning rate for cosine annealing.')
    parser.add_argument('--min-lr', type=str, default='1e-6',
                        help='Min learning rate for cosine annealing.')
    parser.add_argument('--cycle-len', type=int, default='30000',
                        help='Learning rate cycle length for cosine annealing.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    args = parser.parse_args()
    args.epochs = int(args.epochs) if args.epochs else 20
    args.max_lr = float(args.max_lr)
    args.min_lr = float(args.min_lr)
    args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
    args.wd = float(args.wd) if args.wd else 5e-4
    return args


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


def get_dataset(dataset, args):
    if dataset.lower() == 'hico':
        train_dataset = gdata.HICOClassification(split='train', augment_box=False, load_box=True)
        val_dataset = gdata.HICOClassification(split='test', load_box=True, ignore_label=-1)
        val_metric = VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, hico_ap_type=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(HORelationDefaultTrainTransform(net.short, net.max_size)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(HORelationDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
                    epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        cls_scores = []
        gt_classes = []
        for data, label, box in zip(*batch):
            gt_box = label[:, :, :4]
            # get prediction results
            cls_score = net(data, gt_box, box)
            # MIL for image level classification, shape (B, C)
            cls_score = mx.nd.sigmoid(cls_score.max(axis=1, keepdims=False), axis=-1)
            cls_scores.append(cls_score[:, :])
            gt_classes.append(label[:, 0, 5:])

        # update metric
        for score, gt_class in zip(cls_scores, gt_classes):
            eval_metric.update(score, gt_class)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.max_lr,
         'wd': args.wd,
         'momentum': args.momentum,
         'clip_gradient': 5})

    lr_warmup = float(args.lr_warmup)  # avoid int division
    # cosine lr annealing
    lr_annealing = CosineAnnealingSchedule(min_lr=args.min_lr, max_lr=args.max_lr, cycle_length=args.cycle_len)
    epoch_size = len(train_data._dataset)

    rcnn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    metrics = [mx.metric.Loss('RCNN_CrossEntropy'), ]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    # logger.info('Load Trainer State: hcrnboxv1a_resnet50_v1d_voca_0008_0.9102_trainer.state')
    # trainer.load_states("hcrnboxv1a_resnet50_v1d_voca_0008_0.9102_trainer.state")
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            elif i % args.log_interval == 0:
                new_lr = lr_annealing(epoch * epoch_size + i)
                logger.info('[Epoch {} Iteration {}] Set learning rate to {}'.format(epoch, i, new_lr))
                trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            with autograd.record():
                for data, label, box in zip(*batch):
                    gt_label = label[:, 0, 5:]
                    gt_box = label[:, :, :4]
                    cls_pred = net(data, gt_box, box)
                    # losses of rcnn
                    rcnn_loss = rcnn_cls_loss(cls_pred.max(axis=1, keepdims=False), gt_label, None, gt_label * 100)
                    # overall losses
                    losses.append(rcnn_loss.sum())
                    metric_losses[0].append(rcnn_loss.sum())
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
            trainer.step(batch_size)
            # update metrics
            if args.log_interval and not (i + 1) % args.log_interval:
                # msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * batch_size/(time.time()-btic), msg))
                btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time()-tic), msg))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            trainer.save_states('{:s}_{:04d}_{:.4f}_trainer.state'.format(args.save_prefix, epoch, current_map))
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('horelation', args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
