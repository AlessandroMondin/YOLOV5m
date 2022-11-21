import math
import random
import numpy as np
import torch
import os
import imagesize
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import resize_image, xywhn2xyxy, xyxy2xywhn, letterbox
import cv2
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
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentations transformations to be performed with A.Compose
        """
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
            annot_file = "coco_2017_coco128_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "train2017"
        else:
            fname = 'images/val2017'
            annot_file = "coco_2017_val_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_val_txt"

        self.fname = fname

        try:
            self.annotations = pd.read_csv(os.path.join(root_directory, "labels", annot_file),
                                           header=None, index_col=0).sort_values(by=[0])
            self.annotations = self.annotations.head((len(self.annotations)-1))  # just removes last line
        except FileNotFoundError:
            annotations = []
            for img_txt in os.listdir("../datasets/coco128/labels/train2017/"):
                img = img_txt.split(".txt")[0]
                try:
                    w, h = imagesize.get(f"../datasets/coco128/images/train2017/{img}.jpg")
                except FileNotFoundError:
                    continue
                annotations.append([str(img) + ".jpg", h, w])
            self.annotations = pd.DataFrame(annotations)
            self.annotations.to_csv(f"../datasets/coco128/labels/coco_2017_coco128_csv.csv")

        self.len_ann = len(self.annotations)
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations, bs)

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
            annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, self.fname, img_name)).convert("RGB"))

        if self.rect_training:
            bboxes = [ann[1:] for ann in annotations if (ann[2] > 0 and ann[3] > 0)]
            bboxes = torch.tensor(bboxes)
            classes = torch.tensor([ann[0] for ann in annotations])
            labels = torch.cat([classes.unsqueeze(1), bboxes], dim=-1)
            sh, sw = img.shape[0:2]
            img, ratio, pad = letterbox(img, (tg_height, tg_width), auto=False, scaleup=False)
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * sw, ratio[1] * sh, padw=pad[0], padh=pad[1])
            nl = len(labels)
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

            # img = resize_image(img, (int(tg_width), int(tg_height)))
            # bboxes = rescale_bboxes(bboxes, [sw, sh], [tg_width, tg_height])
            # bboxes = [list(bboxes[i]) + [classes[i]] for i in range(len(bboxes))]

        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes if self.rect_training else annotations)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        if len(labels) > 0:
            # bboxes = torch.tensor(bboxes).roll(dims=1, shifts=1)
            # yolo_xywh = coco_to_yolo_tensors(bboxes[..., 1:5], w0=tg_width, h0=tg_height)
            # bboxes[..., 1:] = yolo_xywh
            out_bboxes = torch.zeros((labels.shape[0], 6))
            out_bboxes[..., 1:] = labels
        else:
            out_bboxes = torch.zeros((1, 6))

        out_bboxes[..., 1] -= 1

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), out_bboxes

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations, batch_size):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_bs_{}.csv".format(name, self.len_ann, int(batch_size))
        )

        if os.path.isfile(path):
            print("==> Loading cached annotations for rectangular training on train set")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for rectangular training on train set...")
            annotations["h_w_ratio"] = annotations.iloc[:, 1]/annotations.iloc[:, 2]
            annotations.sort_values(["h_w_ratio"], ascending=True, inplace=True)
            # IMPLEMENT POINT 2 OF WORD DOCUMENT
            for i in range(0, len(annotations), batch_size):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]

                """r = self.default_size / max(size)
                if r != 1:  # if sizes are not equal
                    size = (int(size[0] * r), int(size[1] * r))"""

                #  max_dim = max(size)
                #  max_idx = size.index(max_dim)
                #  size[~max_idx] += 32
                #  sz = random.randrange(int(self.default_size * 0.7), int(self.default_size * 1.3)) // 32 * 32
                #  size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                #  size[max_idx] = sz
                if i + batch_size <= len(annotations):
                    bs = batch_size
                else:
                    bs = len(annotations) - i
                for idx in range(bs):
                    shape = [1, 1]
                    # annotations.iloc[i + idx, 2] = size[0]
                    # annotations.iloc[i + idx, 1] = size[1]
                    batch_h_w = annotations.iloc[i+idx: i+bs+4 , 3].values
                    mini, maxi = np.min(batch_h_w), np.max(batch_h_w)
                    if maxi < 1:
                        shape = [maxi, 1]
                    elif mini > 1:
                        shape = [1, 1 / mini]

                    size = np.ceil(np.array(shape) * self.default_size / 32 + 0).astype(int) * 32


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