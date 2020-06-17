"""Stanford 40 Actions dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
import pickle as pkl
from ..base import VisionDataset
from ..transforms import bbox


class Stanford40Action(VisionDataset):
    """Stanford 40 Actions dataset.

    Parameters
    ----------
    root : str, default '~/data/Stanford40'
        Path to folder storing the dataset.
    split : str, default 'train'
        Candidates can be: 'train', 'test'.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 11 classes are mapped into indices from 0 to 10. We can
        customize it by providing a str to int dict specifying how to map class
        names to indicies. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extreamly large.
    """
    CLASSES = ("applauding", "blowing_bubbles", "brushing_teeth", "cleaning_the_floor", "climbing", "cooking",
               "cutting_trees", "cutting_vegetables", "drinking", "feeding_a_horse", "fishing", "fixing_a_bike",
               "fixing_a_car", "gardening", "holding_an_umbrella", "jumping", "looking_through_a_microscope",
               "looking_through_a_telescope", "playing_guitar", "playing_violin", "pouring_liquid", "pushing_a_cart",
               "reading", "phoning", "riding_a_bike", "riding_a_horse", "rowing_a_boat", "running",
               "shooting_an_arrow", "smoking", "taking_photos", "texting_message", "throwing_frisby",
               "using_a_computer", "walking_the_dog", "washing_dishes", "watching_TV", "waving_hands",
               "writing_on_a_board", "writing_on_a_book")

    def __init__(self, root=os.path.join('~', 'data', 'Stanford40'),
                 split='train', index_map=None, preload_label=True,
                 augment_box=False, load_box=False):
        super(Stanford40Action, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._load_box = load_box
        self._augment_box = augment_box
        self._split = split
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'XMLAnnotations', '{}.xml')
        self._image_path = os.path.join(self._root, 'JPEGImages', '{}.jpg')
        self._box_path = os.path.join(self._root, 'Boxes', '{}.pkl')
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
        """Load individual image indices from split."""
        ids = []
        set_file = os.path.join(self._root, 'ImageSplits', split + '.txt')
        with open(set_file, 'r') as f:
            # remove file extensions
            ids += [line.strip().rsplit(sep='.', maxsplit=1)[0] for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name != 'person':
                continue

            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))

            act_cls_name = obj.find('action').text
            cls_id = self.index_map[act_cls_name]
            cls_array = [0] * len(self.classes)
            cls_array[cls_id] = 1
            anno = [xmin, ymin, xmax, ymax, cls_id]
            anno.extend(cls_array)
            label.append(anno)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
