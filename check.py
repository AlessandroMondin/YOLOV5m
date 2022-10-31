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
from utils.utils import resize_image
from utils.bboxes_utils import rescale_bboxes, coco_to_yolo

def plot_coco(image, boxes):
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

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            box[0],
            box[1],
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.show()


if __name__ == "__main__":

    """anchors = torch.tensor(config.ANCHORS)
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=80, anchors=config.ANCHORS,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    weights_path = os.path.join("ultralytics_files", "yolov5m.pt")

    ordered_dict = torch.load(weights_path, map_location="cpu")

    count_parameters(model)

    check_size(model)

    print("Check and compare these metrics with YOLOV5M of Ultralytics")"""

    img = np.array(Image.open(os.path.join(config.ROOT_DIR, "images","train2017", "000000124516.jpg")).convert("RGB"))
    label_path = os.path.join(os.path.join(config.ROOT_DIR, "annotations", "coco_2017_train_txt", "000000124516" + ".txt"))
    annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
    bboxes = [ann[:-1] for ann in annotations]
    classes = [ann[-1] for ann in annotations]

    annot_bboxes = [[classes[i]-1]+[1]+bboxes[i] for i in range(len(bboxes))]

    plot_coco(img, annot_bboxes)

    new_shape = [480, 96]

    # params: rescale_bboxes(bboxes, [starting_width, starting_height], [ending_width, ending_height])
    bboxes = rescale_bboxes(bboxes, list(img.shape[0:2])[::-1], new_shape)

    """yolo_bboxes = []
    for box in bboxes:
        box = coco_to_yolo(box, image_w=new_shape[0], image_h=new_shape[1])
        for i,coord in enumerate(box):
            if i%2==0:
                box[i] *= new_shape[0]
            else:
                box[i] *= new_shape[1]

        yolo_bboxes.append(box)"""

    img = resize_image(img, new_shape)

    annot_bboxes = [[classes[i]-1] + [1] + bboxes[i] for i in range(len(bboxes))]

    plot_coco(img, annot_bboxes)


