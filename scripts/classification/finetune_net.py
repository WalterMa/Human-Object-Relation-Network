import argparse
import logging
import os
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon.data.vision import transforms
import libgcv
from libgcv import data as gdata
from libgcv import utils as gutils


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer learning on the specific dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='polar',
                        help='Training dataset. Now support polar, voc and coco.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Pre-trained network name")
    parser.add_argument('--pretrained', type=str, default='True',
                        help='True or the path of pre-trained model weights.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='14,20', type=str,
                        help='list of learning rate decay epochs as in str')
    parser.add_argument('--log-interval', type=int, default=10,
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
    args = parser.parse_args()
    return args


def get_dataset(dataset):
    if dataset.lower() == 'polar':
        train_dataset = gdata.POLARClassification(split='train')
        val_dataset = gdata.POLARClassification(split='val')
        eval_metric = mx.metric.Accuracy()
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, eval_metric


def get_dataloader(train_dataset, val_dataset, batch_size, num_workers):
    jitter_param = 0.4
    lighting_param = 0.1
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_train = transforms.Compose([
        transforms.Resize(480),
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        train_dataset.transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='rollover')

    val_data = gluon.data.DataLoader(
        val_dataset.transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, last_batch='keep')

    return train_data, val_data


def get_fix_params_pattern(nework):
    if nework.lower() == 'resnet50v1_b':
        fix_pattern = 'resnetv1b0_conv0|resnetv1b0_layers1|resnetv1b0_down1|resnetv1b0.*batchnorm'
    else:
        fix_pattern = '(?!)'  # not match anything
    return fix_pattern


def train(net, train_data, val_data, eval_metric, ctx, args):
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

    fix_pattern = get_fix_params_pattern(args.network)
    param_dict = net.collect_params(fix_pattern)
    for _, param in param_dict.items():
        param.grad_req = 'null'
    logger.info('Fixed such params for net:\n%s' % param_dict)

    trainer = gluon.Trainer(
        net.collect_params(),
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum})
    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_metric = [0]
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_metric = mx.metric.Loss('CELoss')
    num_batch = len(train_data)
    # Start Training
    for epoch in range(args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        tic = time.time()
        btic = time.time()
        loss_metric.reset()
        eval_metric.reset()

        for i, batch in enumerate(train_data):
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            output_list = []
            loss_list = []
            with autograd.record():
                for data, label in zip(data_list, label_list):
                    output = net(data)
                    output_list.append(output)
                    loss_list.append(loss(output, label))
            autograd.backward(loss_list)
            trainer.step(args.batch_size)

            loss_metric.update(None, loss_list)
            eval_metric.update(label_list, output_list)

            if args.log_interval and not (i + 1) % args.log_interval:
                _, train_loss = loss_metric.get()
                metric_name, metric_value = eval_metric.get()
                speed = args.log_interval * args.batch_size / (time.time() - btic)
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, CELoss=%.3f'.format(
                    epoch, i, speed, train_loss, metric_name, metric_value))
                btic = time.time()

        _, train_loss = loss_metric.get()
        metric_name, metric_value = eval_metric.get()
        if not isinstance(metric_value, (list, tuple)):
            metric_name = [metric_name]
            metric_value = [metric_value]
        metric_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(metric_name, metric_value)])
        logger.info('[Epoch {}] Training cost: {:.3f}, CELoss=%.3f, \n{}'.format(
            epoch, (time.time() - tic), train_loss, metric_msg))

        if not (epoch + 1) % args.val_interval:
            metric_name, metric_value = validate(net, val_data, ctx, eval_metric)
            if not isinstance(metric_value, (list, tuple)):
                metric_name = [metric_name]
                metric_value = [metric_value]
            metric_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(metric_name, metric_value)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, metric_msg))
            current_metric = metric_value[-1]
        else:
            current_metric = 0.

        save_params(net, logger, best_metric, current_metric, epoch, args.save_interval, args.save_prefix)


def validate(net, val_data, ctx, eval_metric):
    for i, batch in enumerate(val_data):
        data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        output_list = [net(data) for data in data_list]
        eval_metric.update(label_list, output_list)
    return eval_metric.get()


def save_params(net, logger, best_metric, current_metric, epoch, save_interval, prefix):
    current_metric = float(current_metric)
    if current_metric > best_metric[0]:
        logger.info('[Epoch {}] Eval Metric {} higher than current best {} saving to {}'.format(
                    epoch, current_metric, best_metric, '{:s}_best.params'.format(prefix)))
        best_metric[0] = current_metric
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix+'_best_metric.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_metric))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_metric)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_metric))


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # Initialize the net with pretrained model
    args.save_prefix += '_'.join((args.network, args.dataset))
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = libgcv.model.get_model(args.network, pretrained=True)
    else:
        net = libgcv.model.get_model(args.network, pretrained=False)
        net.load_parameters(args.pretrained.strip())

    # get dataset
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)
    net.reset_class(train_dataset.classes)
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    # data transform and loader
    train_data, val_data = get_dataloader(train_dataset, val_dataset, args.batch_size, args.num_workers)

    train(net, train_data, val_data, eval_metric, ctx, args)