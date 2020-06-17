"""Pascal VOC Classification evaluation."""
from __future__ import division
import numpy as np
import mxnet as mx


class VOCMultiClsMApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for MultiClassification task

    Parameters:
    ---------
    class_names : list of str
        Required, list of class_name
    """
    def __init__(self, class_names=None, ignore_label=None, hico_ap_type=False, voc_action_type=False):
        assert isinstance(class_names, (list, tuple))
        for name in class_names:
            assert isinstance(name, str), "must provide names as str"
        self.class_names = class_names
        super(VOCMultiClsMApMetric, self).__init__('VOCMeanAP')
        num = len(class_names)
        self.name = list(class_names) + ['mAP']
        self.class_names = class_names
        self.num = num + 1
        self.ignore_label = ignore_label
        self.hico_ap_type = hico_ap_type
        self.voc_action_type = voc_action_type
        self._scores = []
        self._labels = []
        self.reset()

    def reset(self):
        """Clear the internal statistics to initial state."""
        self._scores = []
        self._labels = []
        for i in range(len(self.class_names)):
            self._scores.append([])
            self._labels.append([])

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
        if self.voc_action_type:
            names.append('mAP(Without Other)')
            values = np.append(values, np.mean(values[:-2]))
        return names, values

    def save(self, file_name='result.csv'):
        """Save scores and labels."""
        labels = np.array(self._labels)
        scores = np.array(self._scores)
        res = np.concatenate((labels, scores), axis=0).transpose()
        np.savetxt(file_name, res, fmt='%.6f', delimiter=",")

    def load(self, file_name):
        res = np.loadtxt(file_name, delimiter=',')
        self.load_from_nparray(res[:, 0:len(self.class_names)], res[:, len(self.class_names):])

    def load_from_nparray(self, labels, scores):
        assert scores.shape[1] == len(self.class_names), \
            'Scores must have the shape(N, %d).' % len(self.class_names)
        assert labels.shape[1] == len(self.class_names), \
            'Labels must have the shape(N, %d).' % len(self.class_names)
        self._labels = np.asarray(labels.transpose())
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

        num_class = len(self.class_names)
        # Split in batch axis
        for pred_score, gt_label in zip(*[as_numpy(x) for x in [pred_scores, gt_labels]]):
            # pred_score shape (N, C), gt_label shape (N, C)
            pred_score = pred_score.reshape((-1, num_class))
            gt_label = gt_label.reshape((-1, num_class))
            assert pred_score.shape[0] == gt_label.shape[0], 'Num of scores must be the same with num of ground truths.'
            gt_label = gt_label.astype(int)

            # Iterate over classes
            for i in range(len(self.class_names)):
                single_class_score = pred_score[:, i]
                single_class_label = gt_label[:, i]
                self._scores[i].extend(single_class_score.tolist())
                self._labels[i].extend(single_class_label.tolist())

    def _update(self):
        """ update num_inst and sum_metric """
        ap_list = np.zeros(self.num, dtype=np.float32)
        labels = np.array(self._labels)
        scores = np.array(self._scores)

        for a in range(len(self.class_names)):
            single_class_label = labels[a]
            single_class_score = scores[a]
            if self.ignore_label is not None:
                valid_index = np.where(single_class_label != self.ignore_label)
                single_class_score = single_class_score[valid_index]
                single_class_label = single_class_label[valid_index]
            tp = single_class_label == 1
            npos = np.sum(tp, axis=0)
            fp = single_class_label != 1
            sc = single_class_score
            cat_all = np.vstack((tp, fp, sc)).transpose()
            ind = np.argsort(cat_all[:, 2])
            cat_all = cat_all[ind[::-1], :]
            tp = np.cumsum(cat_all[:, 0], axis=0)
            fp = np.cumsum(cat_all[:, 1], axis=0)

            # # Compute precision/recall
            rec = tp / npos
            prec = np.divide(tp, (fp + tp))
            if self.hico_ap_type:
                ap_list[a] = self._average_precision_hico(rec, prec)
            else:
                ap_list[a] = self._average_precision(rec, prec)
        ap_list[-1] = np.mean(np.nan_to_num(ap_list[:-1]))
        return ap_list

    def _average_precision_hico(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

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
        if rec is None or prec is None:
            return np.nan
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.
        return ap

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
    parser.add_argument('--dataset', type=str, default='voca',
                        help='Results dataset.')
    parser.add_argument('--file', type=str, default='',
                        help='Outputs file path to be loaded. Default will load first file '
                             'ends with outputs.csv in current folder.')
    args = parser.parse_args()

    if args.dataset.lower() == 'hico':
        classes = ('airplane board', 'airplane direct', 'airplane exit', 'airplane fly', 'airplane inspect',
               'airplane load', 'airplane ride', 'airplane sit_on', 'airplane wash', 'airplane no_interaction',
               'bicycle carry', 'bicycle hold', 'bicycle inspect', 'bicycle jump', 'bicycle hop_on', 'bicycle park',
               'bicycle push', 'bicycle repair', 'bicycle ride', 'bicycle sit_on', 'bicycle straddle', 'bicycle walk',
               'bicycle wash', 'bicycle no_interaction', 'bird chase', 'bird feed', 'bird hold', 'bird pet',
               'bird release', 'bird watch', 'bird no_interaction', 'boat board', 'boat drive', 'boat exit',
               'boat inspect', 'boat jump', 'boat launch', 'boat repair', 'boat ride', 'boat row', 'boat sail',
               'boat sit_on', 'boat stand_on', 'boat tie', 'boat wash', 'boat no_interaction', 'bottle carry',
               'bottle drink_with', 'bottle hold', 'bottle inspect', 'bottle lick', 'bottle open', 'bottle pour',
               'bottle no_interaction', 'bus board', 'bus direct', 'bus drive', 'bus exit', 'bus inspect', 'bus load',
               'bus ride', 'bus sit_on', 'bus wash', 'bus wave', 'bus no_interaction', 'car board', 'car direct',
               'car drive', 'car hose', 'car inspect', 'car jump', 'car load', 'car park', 'car ride', 'car wash',
               'car no_interaction', 'cat dry', 'cat feed', 'cat hold', 'cat hug', 'cat kiss', 'cat pet', 'cat scratch',
               'cat wash', 'cat chase', 'cat no_interaction', 'chair carry', 'chair hold', 'chair lie_on',
               'chair sit_on', 'chair stand_on', 'chair no_interaction', 'couch carry', 'couch lie_on', 'couch sit_on',
               'couch no_interaction', 'cow feed', 'cow herd', 'cow hold', 'cow hug', 'cow kiss', 'cow lasso',
               'cow milk', 'cow pet', 'cow ride', 'cow walk', 'cow no_interaction', 'dining_table clean',
               'dining_table eat_at', 'dining_table sit_at', 'dining_table no_interaction', 'dog carry', 'dog dry',
               'dog feed', 'dog groom', 'dog hold', 'dog hose', 'dog hug', 'dog inspect', 'dog kiss', 'dog pet',
               'dog run', 'dog scratch', 'dog straddle', 'dog train', 'dog walk', 'dog wash', 'dog chase',
               'dog no_interaction', 'horse feed', 'horse groom', 'horse hold', 'horse hug', 'horse jump', 'horse kiss',
               'horse load', 'horse hop_on', 'horse pet', 'horse race', 'horse ride', 'horse run', 'horse straddle',
               'horse train', 'horse walk', 'horse wash', 'horse no_interaction', 'motorcycle hold',
               'motorcycle inspect', 'motorcycle jump', 'motorcycle hop_on', 'motorcycle park', 'motorcycle push',
               'motorcycle race', 'motorcycle ride', 'motorcycle sit_on', 'motorcycle straddle', 'motorcycle turn',
               'motorcycle walk', 'motorcycle wash', 'motorcycle no_interaction', 'person carry', 'person greet',
               'person hold', 'person hug', 'person kiss', 'person stab', 'person tag', 'person teach', 'person lick',
               'person no_interaction', 'potted_plant carry', 'potted_plant hold', 'potted_plant hose',
               'potted_plant no_interaction', 'sheep carry', 'sheep feed', 'sheep herd', 'sheep hold', 'sheep hug',
               'sheep kiss', 'sheep pet', 'sheep ride', 'sheep shear', 'sheep walk', 'sheep wash',
               'sheep no_interaction', 'train board', 'train drive', 'train exit', 'train load', 'train ride',
               'train sit_on', 'train wash', 'train no_interaction', 'tv control', 'tv repair', 'tv watch',
               'tv no_interaction', 'apple buy', 'apple cut', 'apple eat', 'apple hold', 'apple inspect', 'apple peel',
               'apple pick', 'apple smell', 'apple wash', 'apple no_interaction', 'backpack carry', 'backpack hold',
               'backpack inspect', 'backpack open', 'backpack wear', 'backpack no_interaction', 'banana buy',
               'banana carry', 'banana cut', 'banana eat', 'banana hold', 'banana inspect', 'banana peel',
               'banana pick', 'banana smell', 'banana no_interaction', 'baseball_bat break', 'baseball_bat carry',
               'baseball_bat hold', 'baseball_bat sign', 'baseball_bat swing', 'baseball_bat throw',
               'baseball_bat wield', 'baseball_bat no_interaction', 'baseball_glove hold', 'baseball_glove wear',
               'baseball_glove no_interaction', 'bear feed', 'bear hunt', 'bear watch', 'bear no_interaction',
               'bed clean', 'bed lie_on', 'bed sit_on', 'bed no_interaction', 'bench inspect', 'bench lie_on',
               'bench sit_on', 'bench no_interaction', 'book carry', 'book hold', 'book open', 'book read',
               'book no_interaction', 'bowl hold', 'bowl stir', 'bowl wash', 'bowl lick', 'bowl no_interaction',
               'broccoli cut', 'broccoli eat', 'broccoli hold', 'broccoli smell', 'broccoli stir', 'broccoli wash',
               'broccoli no_interaction', 'cake blow', 'cake carry', 'cake cut', 'cake eat', 'cake hold', 'cake light',
               'cake make', 'cake pick_up', 'cake no_interaction', 'carrot carry', 'carrot cook', 'carrot cut',
               'carrot eat', 'carrot hold', 'carrot peel', 'carrot smell', 'carrot stir', 'carrot wash',
               'carrot no_interaction', 'cell_phone carry', 'cell_phone hold', 'cell_phone read', 'cell_phone repair',
               'cell_phone talk_on', 'cell_phone text_on', 'cell_phone no_interaction', 'clock check', 'clock hold',
               'clock repair', 'clock set', 'clock no_interaction', 'cup carry', 'cup drink_with', 'cup hold',
               'cup inspect', 'cup pour', 'cup sip', 'cup smell', 'cup fill', 'cup wash', 'cup no_interaction',
               'donut buy', 'donut carry', 'donut eat', 'donut hold', 'donut make', 'donut pick_up', 'donut smell',
               'donut no_interaction', 'elephant feed', 'elephant hold', 'elephant hose', 'elephant hug',
               'elephant kiss', 'elephant hop_on', 'elephant pet', 'elephant ride', 'elephant walk', 'elephant wash',
               'elephant watch', 'elephant no_interaction', 'fire_hydrant hug', 'fire_hydrant inspect',
               'fire_hydrant open', 'fire_hydrant paint', 'fire_hydrant no_interaction', 'fork hold', 'fork lift',
               'fork stick', 'fork lick', 'fork wash', 'fork no_interaction', 'frisbee block', 'frisbee catch',
               'frisbee hold', 'frisbee spin', 'frisbee throw', 'frisbee no_interaction', 'giraffe feed',
               'giraffe kiss', 'giraffe pet', 'giraffe ride', 'giraffe watch', 'giraffe no_interaction',
               'hair_drier hold', 'hair_drier operate', 'hair_drier repair', 'hair_drier no_interaction',
               'handbag carry', 'handbag hold', 'handbag inspect', 'handbag no_interaction', 'hot_dog carry',
               'hot_dog cook', 'hot_dog cut', 'hot_dog eat', 'hot_dog hold', 'hot_dog make', 'hot_dog no_interaction',
               'keyboard carry', 'keyboard clean', 'keyboard hold', 'keyboard type_on', 'keyboard no_interaction',
               'kite assemble', 'kite carry', 'kite fly', 'kite hold', 'kite inspect', 'kite launch', 'kite pull',
               'kite no_interaction', 'knife cut_with', 'knife hold', 'knife stick', 'knife wash', 'knife wield',
               'knife lick', 'knife no_interaction', 'laptop hold', 'laptop open', 'laptop read', 'laptop repair',
               'laptop type_on', 'laptop no_interaction', 'microwave clean', 'microwave open', 'microwave operate',
               'microwave no_interaction', 'mouse control', 'mouse hold', 'mouse repair', 'mouse no_interaction',
               'orange buy', 'orange cut', 'orange eat', 'orange hold', 'orange inspect', 'orange peel', 'orange pick',
               'orange squeeze', 'orange wash', 'orange no_interaction', 'oven clean', 'oven hold', 'oven inspect',
               'oven open', 'oven repair', 'oven operate', 'oven no_interaction', 'parking_meter check',
               'parking_meter pay', 'parking_meter repair', 'parking_meter no_interaction', 'pizza buy', 'pizza carry',
               'pizza cook', 'pizza cut', 'pizza eat', 'pizza hold', 'pizza make', 'pizza pick_up', 'pizza slide',
               'pizza smell', 'pizza no_interaction', 'refrigerator clean', 'refrigerator hold', 'refrigerator move',
               'refrigerator open', 'refrigerator no_interaction', 'remote hold', 'remote point', 'remote swing',
               'remote no_interaction', 'sandwich carry', 'sandwich cook', 'sandwich cut', 'sandwich eat',
               'sandwich hold', 'sandwich make', 'sandwich no_interaction', 'scissors cut_with', 'scissors hold',
               'scissors open', 'scissors no_interaction', 'sink clean', 'sink repair', 'sink wash',
               'sink no_interaction', 'skateboard carry', 'skateboard flip', 'skateboard grind', 'skateboard hold',
               'skateboard jump', 'skateboard pick_up', 'skateboard ride', 'skateboard sit_on', 'skateboard stand_on',
               'skateboard no_interaction', 'skis adjust', 'skis carry', 'skis hold', 'skis inspect', 'skis jump',
               'skis pick_up', 'skis repair', 'skis ride', 'skis stand_on', 'skis wear', 'skis no_interaction',
               'snowboard adjust', 'snowboard carry', 'snowboard grind', 'snowboard hold', 'snowboard jump',
               'snowboard ride', 'snowboard stand_on', 'snowboard wear', 'snowboard no_interaction', 'spoon hold',
               'spoon lick', 'spoon wash', 'spoon sip', 'spoon no_interaction', 'sports_ball block',
               'sports_ball carry', 'sports_ball catch', 'sports_ball dribble', 'sports_ball hit', 'sports_ball hold',
               'sports_ball inspect', 'sports_ball kick', 'sports_ball pick_up', 'sports_ball serve',
               'sports_ball sign', 'sports_ball spin', 'sports_ball throw', 'sports_ball no_interaction',
               'stop_sign hold', 'stop_sign stand_under', 'stop_sign stop_at', 'stop_sign no_interaction',
               'suitcase carry', 'suitcase drag', 'suitcase hold', 'suitcase hug', 'suitcase load', 'suitcase open',
               'suitcase pack', 'suitcase pick_up', 'suitcase zip', 'suitcase no_interaction', 'surfboard carry',
               'surfboard drag', 'surfboard hold', 'surfboard inspect', 'surfboard jump', 'surfboard lie_on',
               'surfboard load', 'surfboard ride', 'surfboard stand_on', 'surfboard sit_on', 'surfboard wash',
               'surfboard no_interaction', 'teddy_bear carry', 'teddy_bear hold', 'teddy_bear hug', 'teddy_bear kiss',
               'teddy_bear no_interaction', 'tennis_racket carry', 'tennis_racket hold', 'tennis_racket inspect',
               'tennis_racket swing', 'tennis_racket no_interaction', 'tie adjust', 'tie cut', 'tie hold',
               'tie inspect', 'tie pull', 'tie tie', 'tie wear', 'tie no_interaction', 'toaster hold',
               'toaster operate', 'toaster repair', 'toaster no_interaction', 'toilet clean', 'toilet flush',
               'toilet open', 'toilet repair', 'toilet sit_on', 'toilet stand_on', 'toilet wash',
               'toilet no_interaction', 'toothbrush brush_with', 'toothbrush hold', 'toothbrush wash',
               'toothbrush no_interaction', 'traffic_light install', 'traffic_light repair',
               'traffic_light stand_under', 'traffic_light stop_at', 'traffic_light no_interaction', 'truck direct',
               'truck drive', 'truck inspect', 'truck load', 'truck repair', 'truck ride', 'truck sit_on', 'truck wash',
               'truck no_interaction', 'umbrella carry', 'umbrella hold', 'umbrella lose', 'umbrella open',
               'umbrella repair', 'umbrella set', 'umbrella stand_under', 'umbrella no_interaction', 'vase hold',
               'vase make', 'vase paint', 'vase no_interaction', 'wine_glass fill', 'wine_glass hold', 'wine_glass sip',
               'wine_glass toast', 'wine_glass lick', 'wine_glass wash', 'wine_glass no_interaction', 'zebra feed',
               'zebra hold', 'zebra pet', 'zebra watch', 'zebra no_interaction')
        eval_metric = VOCMultiClsMApMetric(class_names=classes, ignore_label=-1, hico_ap_type=True)
    elif args.dataset.lower() == 'voca':
        classes = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
                   'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')
        eval_metric = VOCMultiClsMApMetric(class_names=classes, ignore_label=-1, voc_action_type=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(args.dataset))

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
