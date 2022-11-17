import random
import numpy as np
import torch
import os
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.utils import resize_image
from utils.bboxes_utils import rescale_bboxes, iou_width_height, coco_to_yolo, non_max_suppression
from utils.plot_utils import plot_image, cells_to_bboxes
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
            annot_file = "coco_2017_train_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_train_txt"
        else:
            fname = 'images/val2017'
            annot_file = "coco_2017_val_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_val_txt"

        self.fname = fname

        self.annotations = pd.read_csv(os.path.join(root_directory, "annotations", annot_file), header=None)
        self.len_ann = len(self.annotations)
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations, bs)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = self.annotations.iloc[idx, 0]
        h = self.annotations.iloc[idx, 1]
        w = self.annotations.iloc[idx, 2]
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(os.path.join(config.ROOT_DIR, "annotations", self.annot_folder, img_name[:-4] + ".txt"))
        # to avoid an annoying "UserWarning: loadtxt: Empty input file"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, self.fname, img_name)).convert("RGB"))

        if self.rect_training:
            bboxes = [ann[:-1] for ann in annotations]
            classes = [ann[-1] for ann in annotations]
            sh, sw = img.shape[0:2]
            # recasting to int just to make it work on opencv old available version on Sagemaker -.-
            img = resize_image(img, (int(w), int(h)))
            bboxes = rescale_bboxes(bboxes, [sw, sh], [w, h])
            bboxes = [list(bboxes[i]) + [classes[i]] for i in range(len(bboxes))]

        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes if self.rect_training else annotations)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        bboxes = torch.tensor(bboxes).roll(dims=1, shifts=1)
        out_bboxes = torch.zeros((bboxes.shape[0], 6))
        out_bboxes[..., 1:] = bboxes

        return img, out_bboxes

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations, batch_size):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "annotations",
            "adaptive_ann_{}_{}_bs_{}.csv".format(name, self.len_ann, int(batch_size))
        )

        if os.path.isfile(path):
            print("==> Loading cached annotations for rectangular training on train set")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for rectangular training on train set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2]/annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)
            # IMPLEMENT POINT 2 OF WORD DOCUMENT
            for i in range(0, len(annotations), batch_size):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.7), int(self.default_size * 1.3)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + batch_size <= len(annotations):
                    bs = batch_size
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