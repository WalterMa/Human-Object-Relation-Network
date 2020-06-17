import os
import argparse
import numpy as np

CLASSES = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
           'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Network outputs file to VOC official Type.')
    parser.add_argument('--imageset', type=str, default='val',
                        help='VOC Results Imageset, val or test. Default is val.')
    parser.add_argument('--file', type=str, default='',
                        help='Outputs file path to be converted. Default will load first file '
                             'ends with outputs.csv in current folder.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.imageset.lower() == 'val':
        index_file = './voc_val_index.txt'
        save_prefix = 'comp10_action_val_{}.txt'
        jumping_start_pos = 307
    elif args.imageset.lower() == 'test':
        index_file = './voc_test_index.txt'
        save_prefix = 'comp10_action_test_{}.txt'
        jumping_start_pos = 613
    else:
        raise NotImplementedError('ImageSet: {} not implemented.'.format(args.imgset))

    # Load Network outputs
    if args.file.strip():
        output = np.loadtxt(args.file.strip(), delimiter=',')
        print("Load Network outputs from: %s" % args.file.strip())
    else:
        output = None
        for filename in os.listdir('.'):
            if filename.endswith('outputs.csv'):
                output = np.loadtxt(filename, delimiter=',')
                print("Load Network outputs from: %s" % filename)
                break
        if output is None:
            raise FileNotFoundError("Cannot find network outputs file in current dir.")
    out_dim = output.shape[1]

    with open(index_file) as f:
        index = f.read().splitlines()

    for i, cls_name in enumerate(CLASSES):
        if cls_name == 'jumping':
            stat_pos = jumping_start_pos
        else:
            stat_pos = 0
        with open(save_prefix.format(cls_name), 'w') as f:
            f.writelines("%s %.6f\n" % item for item in zip(index[stat_pos:], output[stat_pos:, out_dim - 11 + i]))

    print("Convert Completed!")
