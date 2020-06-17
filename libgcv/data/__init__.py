"""
This module provides data loaders and transfomers for vision datasets.
"""
from . import transforms
from . import batchify
from .pascal_voc.detection import VOCDetection
from .pascal_voc.action import VOCAction
from .mscoco.detection import COCODetection
from .stanford40.detection import Stanford40Action
from .hico.classification import HICOClassification
