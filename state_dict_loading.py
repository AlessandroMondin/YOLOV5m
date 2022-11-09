import torch
from model import YOLOV5m
import config
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
from utils.bboxes_utils import non_max_suppression
from utils.plot_utils import cells_to_bboxes, plot_image

# ULTRALYTICS NOT USED
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == "__main__":

    nc = 80
    anchors = config.ANCHORS
    first_out = 48
    S = [8, 16, 32]

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    pretrained_weights = torch.load("ultralytics_files/yolov5m_real.pt")

    pt_values = pretrained_weights.values()

    # Manually loading ultralytics weights in my architecture
    state_dict = model.state_dict()
    layers_loaded = []
    num_layers_loaded = 0
    for idx, (layer, weight) in enumerate(pretrained_weights.items()):
        for my_layer, my_weight in state_dict.items():
            if weight.shape == my_weight.shape:
                if my_layer not in layers_loaded:
                    state_dict[my_layer] = weight
                    num_layers_loaded += 1
                    layers_loaded.append(my_layer)
                    break

    # print(num_layers_loaded)
    # print(len(layers_loaded))

    equal_layers = 0
    state_dict_values = list(state_dict.values())

    for idx, (key, value) in enumerate(pretrained_weights.items()):
        if torch.equal(value.float(), state_dict_values[idx].float()):
            equal_layers += 1

    # print(equal_layers)

    # torch.save(state_dict, "yolov5_my_arch_ultra_w.pt")
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    img = np.array(Image.open("ultralytics_files/test_images/zidane.jpg").convert("RGB"))
    img = transforms.ToTensor()(img)
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    tg_size = (384, 640)
    img = transforms.Resize(tg_size, interpolation=transforms.InterpolationMode.NEAREST)(img)
    with torch.no_grad():
        out = model(img)

    boxes = cells_to_bboxes(out, model.head.anchors, S, list_output=False, is_pred=True)
    boxes = non_max_suppression(boxes, iou_threshold=0.6, threshold=.25, max_detections=300)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), boxes[0])
