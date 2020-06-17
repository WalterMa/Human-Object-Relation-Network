"""Pascal VOC Classification evaluation."""
from __future__ import division
import numpy as np
import mxnet as mx


class VOCClsMApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for classification task

    Parameters:
    ---------
    class_names : list of str
        Required, list of class_name
    ignore_label : int or None
        None to disable filter label, else will exclude the result
        when its label equals to ignore_label
    """
    def __init__(self, class_names=None, ignore_label=-1):
        assert isinstance(class_names, (list, tuple))
        for name in class_names:
            assert isinstance(name, str), "must provide names as str"
        self.class_names = class_names
        super(VOCClsMApMetric, self).__init__('VOCMeanAP')
        num = len(class_names)
        self.name = list(class_names) + ['mAP']
        self.class_names = class_names
        self.num = num + 1
        self.ignore_label = ignore_label
        self._scores = []
        self._labels = []
        self.reset()

    def reset(self):
        """Clear the internal statistics to initial state."""
        self._scores = []
        self._labels = []
        for i in range(len(self.class_names)):
            self._scores.append([])

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : numpy.float32
           Value of the evaluation.
        """
        names = ['%s' % (self.name[i]) for i in range(self.num)]
        values = self._update()
        return names, values

    def save(self, file_name='result.csv'):
        """Save scores and labels."""
        labels = np.array(self._labels).reshape((1, -1))
        scores = np.array(self._scores)
        res = np.concatenate((labels, scores), axis=0).transpose()
        np.savetxt(file_name, res, fmt='%.6f', delimiter=",")

    def load(self, file_name):
        res = np.loadtxt(file_name, delimiter=',')
        self.load_from_nparray(res[:, 0], res[:, 1:])

    def load_from_nparray(self, labels, scores):
        assert scores.shape[1] == len(self.class_names), \
            'Scores must have the shape(N, %d).' % len(self.class_names)
        assert labels.shape[0] == scores.shape[0], \
            'Labels num: %d is not equal to scores num: %d.' % (labels.shape[0], scores.shape[0])
        self._labels = np.asarray(np.squeeze(labels))
        self._scores = np.asarray(scores.transpose())

    def update(self, pred_scores, gt_labels):
        """Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction class scores with shape `B, N, C`.
        gt_labels : mxnet.NDArray or numpy.ndarray
            Ground-truth labels with shape `B, N`.
        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                out = np.array(out)
                # just return out directly for 1-d array
                if len(out.shape) == 1:
                    return out
                return np.concatenate(out, axis=0)
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        # Split in batch axis
        for pred_score, gt_label in zip(*[as_numpy(x) for x in [pred_scores, gt_labels]]):
            # pred_score shape (N, C), gt_label shape (N, )
            num_class = len(self.class_names)
            assert pred_score.shape[1] == num_class, 'Predicted Scores %s must have the shape(B, N, %d).' % \
                                                     (str(pred_scores.shape), num_class)
            assert pred_score.shape[0] == gt_label.shape[0], 'Num of scores must be the same with num of ground truths.'
            gt_label = gt_label.flatten().astype(int)
            if self.ignore_label is not None:
                valid_index = np.where(gt_label != self.ignore_label)[0]
                pred_score = pred_score[valid_index]
                gt_label = gt_label[valid_index]

            # Iterate over classes
            for i in range(len(self.class_names)):
                self._scores[i].extend(pred_score[:, i].tolist())
            self._labels.extend(gt_label.tolist())

    def _update(self):
        """ update num_inst and sum_metric """
        ap_list = np.zeros(self.num, dtype=np.float32)
        labels = np.array(self._labels)
        scores = np.array(self._scores)

        for a in range(len(self.class_names)):
            tp = labels == a
            npos = np.sum(tp, axis=0)
            fp = labels != a
            sc = scores[a]
            cat_all = np.vstack((tp, fp, sc)).transpose()
            ind = np.argsort(cat_all[:, 2])
            cat_all = cat_all[ind[::-1], :]
            tp = np.cumsum(cat_all[:, 0], axis=0)
            fp = np.cumsum(cat_all[:, 1], axis=0)

            # # Compute precision/recall
            rec = tp / npos
            prec = np.divide(tp, (fp + tp))
            ap_list[a] = self._average_precision(rec, prec)
        ap_list[-1] = np.mean(ap_list[:-1])
        return ap_list

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        rec = rec.reshape(rec.size, 1)
        prec = prec.reshape(prec.size, 1)
        z = np.zeros((1, 1))
        o = np.ones((1, 1))
        mrec = np.vstack((z, rec, o))
        mpre = np.vstack((z, prec, z))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # i = find(mrec(2:end)~=mrec(1:end-1))+1;
        I = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = 0
        for i in I:
            ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Load and eval outputs file.')
    parser.add_argument('--dataset', type=str, default='st40',
                        help='Results dataset, voca or st40. Default is voca.')
    parser.add_argument('--file', type=str, default='',
                        help='Outputs file path to be loaded. Default will load first file '
                             'ends with outputs.csv in current folder.')
    args = parser.parse_args()

    if args.dataset.lower() == 'voca':
        classes = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
                   'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')
    elif args.dataset.lower() == 'st40':
        classes = ("applauding", "blowing_bubbles", "brushing_teeth", "cleaning_the_floor", "climbing", "cooking",
                   "cutting_trees", "cutting_vegetables", "drinking", "feeding_a_horse", "fishing", "fixing_a_bike",
                   "fixing_a_car", "gardening", "holding_an_umbrella", "jumping", "looking_through_a_microscope",
                   "looking_through_a_telescope", "playing_guitar", "playing_violin", "pouring_liquid",
                   "pushing_a_cart",
                   "reading", "phoning", "riding_a_bike", "riding_a_horse", "rowing_a_boat", "running",
                   "shooting_an_arrow", "smoking", "taking_photos", "texting_message", "throwing_frisby",
                   "using_a_computer", "walking_the_dog", "washing_dishes", "watching_TV", "waving_hands",
                   "writing_on_a_board", "writing_on_a_book")
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(args.dataset))
    eval_metric = VOCClsMApMetric(class_names=classes)

    # Load Network outputs
    if args.file.strip():
        print("Load Network outputs from: %s" % args.file.strip())
        eval_metric.load(args.file.strip())
    else:
        hasOutputFile = False
        for filename in os.listdir('.'):
            if filename.endswith('outputs.csv'):
                hasOutputFile = True
                print("Load Network outputs from: %s" % filename)
                eval_metric.load(filename)
                break
        if not hasOutputFile:
            raise FileNotFoundError("Cannot find network outputs file in current dir.")

    map_name, mean_ap = eval_metric.get()
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    print('Results: \n{}'.format(val_msg))
