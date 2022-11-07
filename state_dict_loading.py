import torch
from model import YOLOV5m
import config

from PIL import Image
from torchvision import transforms
from utils.bboxes_utils import non_max_suppression
from utils.plot_utils import cells_to_bboxes, plot_image

if __name__ == "__main__":

    nc = 80
    anchors = torch.tensor(config.ANCHORS)
    first_out = 48
    S = [8, 16, 32]

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    pretrained_weights = torch.load("ultralytics_files/yolov5m_state_dict.pt")

    state_dict = model.state_dict()
    layers_loaded = []
    num_layers_loaded = 0
    for idx, (layer, weight) in enumerate(pretrained_weights.items()):
        for my_layer, my_weight in state_dict.items():
            if weight.shape == my_weight.shape:
                if my_layer not in layers_loaded:
                    print(my_layer, "---", layer)
                    state_dict[my_layer] = weight
                    num_layers_loaded += 1
                    layers_loaded.append(my_layer)
                    break

    print(num_layers_loaded)
    print(len(layers_loaded))

    equal_layers = 0
    state_dict_values = list(state_dict.values())

    for idx, (key, value) in enumerate(pretrained_weights.items()):
        if torch.equal(value.float(), state_dict_values[idx].float()):
            equal_layers += 1

    print(equal_layers)

    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()
    img = torch.unsqueeze(transforms.ToTensor()(Image.open("ultralytics_files/test_images/zidane.jpg").convert("RGB")),0)
    tg_size = (384, 640)
    #tg_size = (img.shape[2]//32*32, img.shape[3]//32*32)
    img = transforms.Resize(tg_size)(img)

    out = model(img)
    boxes = cells_to_bboxes(out, model.head.anchors, S, list_output=False, is_pred=True)
    boxes = non_max_suppression(boxes, iou_threshold=0.6, threshold=.25, max_detections=300)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), boxes[0])
