import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import imagesize
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import resize_image, xyxy2xywhn
from utils.bboxes_utils import rescale_bboxes, coco_to_yolo_tensors
import config


class MS_COCO_2017(Dataset):
    """COCO 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 num_classes,
                 anchors,
                 root_directory=config.ROOT_DIR,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 rect_training=False,
                 default_size=640,
                 bs=64,
                 bboxes_format="coco",
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentations transformations to be performed with A.Compose
        """
        assert bboxes_format in ["coco", "yolo"], 'bboxes_format must be either "coco" or "yolo"'

        self.batch_range = 64 if bs < 64 else 128

        self.bboxes_format = bboxes_format
        self.nc = num_classes
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.S = S
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.rect_training = rect_training
        self.default_size = default_size
        self.root_directory = root_directory
        self.train = train

        if train:
            fname = 'images/train2017'
            annot_file = "annot_train.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "train2017"
        else:
            fname = 'images/val2017'
            annot_file = "annot_val.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "val2017"

        self.fname = fname

        try:
            self.annotations = pd.read_csv(os.path.join(root_directory, "labels", annot_file),
                                           header=None, index_col=0).sort_values(by=[0])
            self.annotations = self.annotations.head((len(self.annotations)-1))  # just removes last line
        except FileNotFoundError:
            annotations = []
            for img_txt in os.listdir(os.path.join(self.root_directory, "labels", self.annot_folder)):
                img = img_txt.split(".txt")[0]
                try:
                    w, h = imagesize.get(os.path.join(self.root_directory, "images", self.annot_folder, f"{img}.jpg"))
                except FileNotFoundError:
                    continue
                annotations.append([str(img) + ".jpg", h, w])
            self.annotations = pd.DataFrame(annotations)
            self.annotations.to_csv(os.path.join(self.root_directory, "labels", annot_file))

        self.len_ann = len(self.annotations)
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = self.annotations.iloc[idx, 0]
        tg_height = self.annotations.iloc[idx, 1] if self.rect_training else 640
        tg_width = self.annotations.iloc[idx, 2] if self.rect_training else 640
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(self.root_directory, "labels", self.annot_folder, img_name[:-4] + ".txt")
        # to avoid an annoying "UserWarning: loadtxt: Empty input file"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
            # removing annotations with negative values
            labels = labels[np.all(labels >= 0, axis=1), :]
            # to avoid negative values
            labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, self.fname, img_name)).convert("RGB"))

        if self.bboxes_format == "coco":
            labels[:, -1] -= 1  # 0-indexing the classes of coco labels (1-80 --> 0-79)
            labels = np.roll(labels, axis=1, shift=1)
            # normalized coordinates are scale invariant, hence after resizing the img we don't resize labels
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], w0=img.shape[1], h0=img.shape[0])

        img = resize_image(img, (int(tg_width), int(tg_height)))

        if self.transform:
            if len(labels) > 0:
                # albumentations requires bboxes to be (x,y,w,h,class_idx)
                augmentations = self.transform(image=img,
                                               bboxes=np.roll(labels, axis=1, shift=4)
                                               )
                img = augmentations["image"]
                # loss fx requires bboxes to be (class_idx,x,y,w,h)
                labels = np.array(augmentations["bboxes"])

        if len(labels) > 0:
            labels = np.roll(labels, axis=1, shift=1)
            labels = torch.from_numpy(labels)
            out_bboxes = torch.zeros((labels.shape[0], 6))
            out_bboxes[..., 1:] = labels
        else:
            out_bboxes = torch.zeros((1, 6))

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), out_bboxes

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_br_{}.csv".format(name, self.len_ann, int(self.batch_range))
        )

        if os.path.isfile(path):
            print("==> Loading cached annotations for rectangular training on val set")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for 'rectangular training' on training set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2] / annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)

            for i in range(0, len(annotations), self.batch_range):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.9), int(self.default_size * 1.1)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + self.batch_range <= len(annotations):
                    bs = self.batch_range
                else:
                    bs = len(annotations) - i
                for idx in range(bs):
                    annotations.iloc[i + idx, 2] = size[0]
                    annotations.iloc[i + idx, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:idx, :] = annotations.iloc[i:idx, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations.iloc[:,:3])
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)