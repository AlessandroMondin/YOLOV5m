from collections import OrderedDict
import torch
from model import YOLOV5m
import config
import numpy as np

from PIL import Image
from torchvision import transforms
from utils.bboxes_utils import non_max_suppression
from utils.plot_utils import cells_to_bboxes, plot_image


if __name__ == "__main__":

    nc = 80
    anchors = config.ANCHORS
    first_out = 48
    S = [8, 16, 32]

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16))

    pretrained_weights = torch.load("ultralytics_yolov5m.pt")

    pt_values = pretrained_weights.values()
    """
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
    torch.save(state_dict, "../yolov5m.pt")
    model.load_state_dict(torch.load("../yolov5m.pt"))

    state_dict = model.state_dict()
    car_person_heads = []
    for key, vals in state_dict.items():
        if "head" in key and "anchors" not in key:
            if len(vals.shape) > 1:
                layer_1 = torch.cat([vals[0:5, :, :, :], vals[7:8, :, :, :], vals[5:6, :, :, :]], dim=0)
                layer_2 = torch.cat([vals[85:90, :, :, :], vals[92:93, :, :, :], vals[90:91, :, :, :]], dim=0)
                layer_3 = torch.cat([vals[170:175, :, :, :], vals[177:178, :, :, :], vals[175:176, :, :, :]], dim=0)
                car_person_heads.append([key, torch.cat([layer_1, layer_2, layer_3], dim=0)])
            else:
                layer_1 = torch.cat([vals[0:5], vals[7:8], vals[5:6]], dim=0)
                layer_2 = torch.cat([vals[85:90], vals[92:93], vals[90:91]], dim=0)
                layer_3 = torch.cat([vals[170:175], vals[177:178], vals[175:176]], dim=0)
                car_person_heads.append([key, torch.cat([layer_1, layer_2, layer_3], dim=0)])

        else:
            car_person_heads.append([key, vals])

    state_dict_cp = OrderedDict(car_person_heads)
    torch.save(OrderedDict(car_person_heads), "../yolov5m_nh.pt")

    no_heads = []
    for key, vals in state_dict.items():
        if "head" in key and "anchors" not in key:
            continue
        else:
            no_heads.append([key, vals])

    state_dict_no_heads = OrderedDict(no_heads)

    torch.save(state_dict_no_heads, "../yolov5m_nh.pt")
    """
    model = YOLOV5m(first_out=first_out, nc=80, anchors=anchors,
                    ch=(first_out * 4, first_out * 8, first_out * 16))

    model.load_state_dict(state_dict=torch.load("../yolov5m.pt"), strict=True)
    model.eval()

    img = np.array(Image.open("test_images/hollywood.jpg").convert("RGB"))
    img = transforms.ToTensor()(img)
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    tg_size = (480, 736)
    img = transforms.Resize(tg_size, interpolation=transforms.InterpolationMode.NEAREST)(img)
    with torch.no_grad():
        out = model(img)
    boxes = cells_to_bboxes(out, model.head.anchors, S, list_output=False, is_pred=True)
    boxes = non_max_suppression(boxes, iou_threshold=0.6, threshold=.25, max_detections=300)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), boxes[0])
