"""HICO Action classification dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
import json
import mxnet as mx
import pickle as pkl
from ..base import VisionDataset
from ..transforms import bbox


class HICOClassification(VisionDataset):
    """HICO Classification.

    Parameters
    ----------
    root : str, default '~/data/hico'
        Path to folder storing the dataset.
    split : str, default 'train'
        Candidates can be: 'train', 'test'.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 9 classes are mapped into indices from 0 to 8. We can
        customize it by providing a str to int dict specifying how to map class
        names to indicies. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extreamly large.
    """
    CLASSES = ['airplane board', 'airplane direct', 'airplane exit', 'airplane fly', 'airplane inspect',
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
               'zebra hold', 'zebra pet', 'zebra watch', 'zebra no_interaction']

    def __init__(self, root=os.path.join('~', 'data', 'hico'),
                 split='train', index_map=None, preload_label=True,
                 augment_box=False, load_box=False, ignore_label=None, random_cls=False):
        super(HICOClassification, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._load_box = load_box
        self._augment_box = augment_box
        self._split = split
        self._ignore_label = ignore_label
        self._random_cls = random_cls
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'annotations', '{}.json')
        self._image_path = os.path.join(self._root, 'images', '{}.jpg')
        self._box_path = os.path.join(self._root, 'boxes', '{}.pkl')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        return self.__class__.__name__ + '(' + self._split + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def img_path(self, idx):
        img_id = self._items[idx]
        return self._image_path.format(img_id)

    def save_boxes(self, idx, boxes):
        img_id = self._items[idx]
        box_path = self._box_path.format(img_id)
        with open(box_path, 'wb') as f:
            pkl.dump(boxes, f)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._random_cls:
            for i, cls in enumerate(label[:, 5:]):
                candidate_cls = np.array(np.where(cls == 1)).reshape((-1,))
                label[i, 4] = np.random.choice(candidate_cls)
        if self._augment_box:
            h, w, _ = img.shape
            label = bbox.augment(label, img_w=w, img_h=h, output_num=16)
        if self._load_box:
            box_path = self._box_path.format(img_id)
            with open(box_path, 'rb') as f:
                box = pkl.load(f)
            return img, label, box
        return img, label

    def _load_items(self, split):
        """Load individual image indices from splits."""
        ids = []
        set_file = os.path.join(self._root, 'sets', split + '.txt')
        with open(set_file, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse json file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(img_id)
        with open(anno_path, 'r') as f:
            anno = json.load(f)
            width = float(anno['width'])
            height = float(anno['height'])
            if idx not in self._im_shapes:
                # store the shapes for later usage
                self._im_shapes[idx] = (width, height)
            label = []
            cls = anno['actions']
            pos_cls_names = [key for key, value in cls.items() if value == 1]
            cls_id = self.index_map[pos_cls_names[0]]
            cls_array = [0] * len(self.classes)
            for n in pos_cls_names:
                cls_array[self.index_map[n]] = 1
            if self._ignore_label is not None:
                ignore_cls_names = [key for key, value in cls.items() if value == 0]
                for n in ignore_cls_names:
                    cls_array[self.index_map[n]] = int(self._ignore_label)
            for p in anno['persons']:
                xmin = p['bndbox']['xmin']
                xmax = p['bndbox']['xmax']
                ymin = p['bndbox']['ymin']
                ymax = p['bndbox']['ymax']
                try:
                    self._validate_label(xmin, ymin, xmax, ymax, width, height)
                except AssertionError as e:
                    raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
                anno = [xmin, ymin, xmax, ymax, cls_id]
                anno.extend(cls_array)
                label.append(anno)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert xmin >= 0 and xmin < width, (
            "xmin must in [0, {}), given {}".format(width, xmin))
        assert ymin >= 0 and ymin < height, (
            "ymin must in [0, {}), given {}".format(height, ymin))
        assert xmax > xmin and xmax <= width, (
            "xmax must in (xmin, {}], given {}".format(width, xmax))
        assert ymax > ymin and ymax <= height, (
            "ymax must in (ymin, {}], given {}".format(height, ymax))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
