from pathlib import Path
import cv2 
import torch 
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_imshow, non_max_suppression,apply_classifier, xyxy2xywh, scale_coords, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class peopleDetect:
    def __init__(self, source, weights, device):
        self.device = select_device(device)
        self.model = attempt_load(weights=weights, map_location=device)