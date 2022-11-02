import os.path

import matplotlib.pyplot as plt
import config
import numpy as np
import matplotlib.patches as patches
import torch
from utils.bboxes_utils import non_max_suppression as nms


# ALADDIN'S
def cells_to_bboxes_aladdin(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = box_predictions[..., 0:2].sigmoid() * 2 - 0.5
        box_predictions[..., 2:4] = (box_predictions[..., 2:4].sigmoid() * 2) ** 2 * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
    )

    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    scale_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return scale_bboxes.tolist()


def cells_to_bboxes(predictions, anchors, strides, is_pred=False, list_output=True):
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grid(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)
        if is_pred:
            predictions[i] = predictions[i].to(config.DEVICE)
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            obj = torch.sigmoid(predictions[i][..., 0:1])
            xy = (2 * (predictions[i][..., 1:3] - 0.5) + grid[i]) * stride
            wh = ((2*predictions[i][..., 3:5])**2) * anchor_grid[i] * stride
            best_class = torch.argmax(predictions[i][..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to(config.DEVICE)
            obj = predictions[i][..., 0:1]
            xy = (predictions[i][..., 1:3] + grid[i]) * stride
            wh = predictions[i][..., 3:5] * stride  # * anchor_grid[i]
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if list_output else torch.cat(all_bboxes, dim=1)


def make_grid(anchors, naxs, stride, nx=20, ny=20, i=0, pred=False):
    d = anchors[i].device
    t = anchors[i].dtype
    shape = 1, naxs, ny, nx, 2  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xv, yv), 2).expand(shape)  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (anchors[i] * stride).view((1, naxs, 1, 1, 2)).expand(shape)

    return grid, anchor_grid


def save_predictions(model, loader, folder, epoch, device, filename, num_images=10):

    print("=> Saving images predictions...")

    if not os.path.exists(path=os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch + 1)}')):
        os.makedirs(os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch + 1)}'))

    path = os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch + 1)}')
    anchors = model.head.anchors

    model.eval()

    for idx, (images, targets) in enumerate(loader):

        images = images.to(device)

        if idx < num_images:
            with torch.no_grad():


                out = model(images)
                boxes = cells_to_bboxes(out, anchors, model.head.stride, is_pred=True)[0]
                gt_boxes = cells_to_bboxes(targets, anchors, model.head.stride, is_pred=False)[0]
                
                # here using different nms_iou_thresh and config_thresh because of 
                # https://github.com/ultralytics/yolov5/issues/4464
                boxes = nms(boxes, iou_threshold=0.45, threshold=0.25, box_format="midpoint")
                gt_boxes = nms(gt_boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")

                cmap = plt.get_cmap("tab20b")
                class_labels = config.COCO_LABELS
                colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
                im = np.array(images[0].permute(1, 2, 0).cpu())

                # Create figure and axes
                fig, (ax1, ax2) = plt.subplots(1, 2)
                # Display the image
                ax1.imshow(im)
                ax2.imshow(im)

                # box[0] is x midpoint, box[2] is width
                # box[1] is y midpoint, box[3] is height
                axes = [ax1, ax2]
                # Create a Rectangle patch
                boxes = [gt_boxes, boxes]
                for i in range(2):
                    for box in boxes[i]:
                        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
                        class_pred = int(box[0])

                        box = box[2:]
                        upper_left_x = max(box[0] - box[2] / 2, 0)
                        upper_left_x = min(upper_left_x, im.shape[1])
                        lower_left_y = max(box[1] - box[3] / 2, 0)
                        lower_left_y = min(lower_left_y, im.shape[0])

                        # print(upper_left_x)
                        # print(lower_left_y)
                        rect = patches.Rectangle(
                            (upper_left_x, lower_left_y),
                            box[2],
                            box[3],
                            linewidth=2,
                            edgecolor=colors[class_pred],
                            facecolor="none",
                        )
                        # Add the patch to the Axes
                        if i == 0:
                            axes[i].set_title("Ground Truth bboxes")
                        else:
                            axes[i].set_title("Predicted bboxes")
                        axes[i].add_patch(rect)
                        axes[i].text(
                            upper_left_x,
                            lower_left_y,
                            s=class_labels[class_pred],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": colors[class_pred], "pad": 0},
                            fontsize="small"
                        )

                fig.savefig(f'{path}/image_{idx}.png', dpi=300)
                plt.close(fig)
        # if idx > num images
        else:
            break

    model.train()


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]

        # FOR MY_NMS attempts, also rect = patches.Rectangle box[2] becomes box[2] - box[0] and box[3] - box[1]
        """upper_left_x = max(box[0], 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(box[1], 0)
        lower_left_y = min(lower_left_y, im.shape[0])"""

        upper_left_x = max(box[0] - box[2] / 2, 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(box[1] - box[3] / 2, 0)
        lower_left_y = min(lower_left_y, im.shape[0])

        rect = patches.Rectangle(
            (upper_left_x, lower_left_y),
            box[2],
            box[3],
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            lower_left_y,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.show()
