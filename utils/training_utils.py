import torch
import torch.nn as nn
import random
import math
from tqdm import tqdm
import config
from torch.utils.data import DataLoader

import model
from utils.bboxes_utils import rescale_bboxes
from dataset import MS_COCO_2017, MS_COCO_2017_VALIDATION


def multi_scale(img, target_shape, max_stride):
    # to make it work with collate_fn of the loader
    # returns a random number between target_shape*0.5 e target_shape*1.5+max_stride, applies an integer
    # division by max stride and multiplies again for max_stride
    # in other words it returns a number between those two interval divisible by 32
    sz = random.randrange(target_shape * 0.5, target_shape + max_stride) // max_stride * max_stride
    # sf is the ratio between the random number and the max between height and width
    sf = sz / max(img.shape[2:])
    h, w = img.shape[2:]
    # 1) regarding the larger dimension (height or width) it will become the closest divisible by 32 of
    # larger_dimension*sz
    # 2) regarding the smaller dimension (height or width) it will become the closest divisible by 32 of
    # smaller_dimension*sf (random_number_divisible_by_32_within_range/larger_dimension)
    # math.ceil is the opposite of floor, it rounds the floats to the next ints
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]]
    # ns are the height,width that the new image will have
    imgs = nn.functional.interpolate(img, size=ns, mode="bilinear", align_corners=False)
    return imgs


def get_loaders(
        db_root_dir,
        batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        rect_training=False,
        coco128val=False
):

    S = [8, 16, 32]

    train_augmentation = config.ADAPTIVE_TRAIN_TRANSFORM if rect_training else config.TRAIN_TRANSFORMS
    val_augmentation = config.ADAPTIVE_VAL_TRANSFORM if rect_training else config.VAL_TRANSFORM

    # bs here is not batch_size, check class method "adaptive_shape" to check behavior
    train_ds = MS_COCO_2017(num_classes=len(config.COCO80), anchors=config.ANCHORS,
                            root_directory=db_root_dir, transform=train_augmentation,
                            train=True, S=S, rect_training=rect_training, bs=64,
                            default_size=640)

    val_ds = MS_COCO_2017_VALIDATION(num_classes=len(config.COCO80), anchors=config.ANCHORS,
                                     root_directory=db_root_dir, transform=val_augmentation,
                                     train=False, S=S, rect_training=rect_training, bs=64,
                                     default_size=640, coco_128=coco128val)

    shuffle = False if rect_training else True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=train_ds.collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=None
    )

    return train_loader, val_loader


# define train_loop
def train_loop(model, loader, optim, loss_fn, scaler, epoch, num_epochs, multi_scale_training=True):
    print(f"Training epoch {epoch}/{num_epochs}")
    # these first 4 rows are copied from Ultralytics repo. They studied a mechanism to make model's learning
    # batch_invariant for bs between 1 and 64: based on batch_size the loss is cumulated in the scaler (optimizer) but the
    # frequency of scaler.step() (optim.step()) depends on the batch_size
    # check here: https://github.com/ultralytics/yolov5/issues/2377
    nbs = 64  # nominal batch size
    batch_size = next(iter(loader))[0][0].shape[0]
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    last_opt_step = -1

    loop = tqdm(loader)
    avg_batches_loss = 0
    loss_epoch = 0
    nb = len(loader)
    optim.zero_grad()
    for idx, (images, bboxes) in enumerate(loop):
        images = torch.stack(images, dim=0)

        if multi_scale_training:
            images = multi_scale(images, target_shape=640, max_stride=32)

        images = images.to(config.DEVICE, non_blocking=True)
        # BBOXES AND CLASSES ARE PUSHED to.(DEVICE) INSIDE THE LOSS_FN

        # float16 training: reduces the load inside the VRAM and speeds up the training
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out, bboxes, pred_size=images.shape[2:4], batch_idx=idx, epoch=epoch)
            avg_batches_loss += loss

        # backpropagation
        # check docs here https://pytorch.org/docs/stable/amp.html
        scaler.scale(loss).backward()

        if idx - last_opt_step >= accumulate or (idx == nb-1):
            scaler.unscale_(optim)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optim)  # optimizer.step
            scaler.update()
            optim.zero_grad(set_to_none=True)
            last_opt_step = idx

        # update tqdm loop
        freq = 10
        if idx % freq == 0:
            loop.set_postfix(average_loss_batches=avg_batches_loss.item() / freq)
            avg_batches_loss = 0

    print(
        f"==> training_loss: {loss_epoch / len(loader):2f}"
    )
