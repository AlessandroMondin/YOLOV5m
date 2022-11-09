import torch
from utils.utils import check_size, count_parameters
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import YOLOV5m
import config
from PIL import Image
import numpy as np
import cv2
from utils.plot_utils import plot_image
from utils.utils import resize_image, coco91_2_coco80
from utils.bboxes_utils import rescale_bboxes, coco_to_yolo

print(coco91_2_coco80())



