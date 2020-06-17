"""Pascal VOC object detection dataset."""
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


class VOCAction(VisionDataset):
    """Pascal VOC2012 Action Dataset.

    Parameters
    ----------
    root : str, default '~/data/VOCdevkitc'
        Path to folder storing the dataset.
    split : str, default 'train'
        Candidates can be: 'train', 'val', 'trainval', 'test'.
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
    CLASSES = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
               'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')

    def __init__(self, root=os.path.join('~', 'data', 'VOCdevkit'),
                 split='train', index_map=None, preload_label=True,
                 augment_box=False, load_box=False, random_cls=False):
        super(VOCAction, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.join(os.path.expanduser(root), 'VOC2012')
        self._augment_box = augment_box
        self._load_box = load_box
        self._random_cls = random_cls
        self._split = split
        if self._split.lower() == 'val':
            self._jumping_start_pos = 307
        elif self._split.lower() == 'test':
            self._jumping_start_pos = 613
        else:
            self._jumping_start_pos = 0
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'Annotations', '{}.xml')
        self._box_path = os.path.join(self._root, 'Boxes', '{}.pkl')
        self._image_path = os.path.join(self._root, 'JPEGImages', '{}.jpg')
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
        """Load individual image indices from split."""
        ids = []
        set_file = os.path.join(self._root, 'ImageSets', 'Action', split + '.txt')
        with open(set_file, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
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

            cls_id = -1
            act_cls = obj.find('actions')
            cls_array = [0] * len(self.classes)
            if idx < self._jumping_start_pos:
                # ignore jumping class according to voc offical code
                cls_array[0] = -1
            if act_cls is not None:
                for i, cls_name in enumerate(self.classes):
                    is_action = float(act_cls.find(cls_name).text)
                    if is_action > 0.5:
                        cls_id = i
                        cls_array[i] = 1
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
