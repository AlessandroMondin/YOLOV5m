import torch

import sys
sys.path.insert(0, '../yolov5')


from model import YOLOV5m
import config


if __name__ == "__main__":

    batch_size = 2
    image_height = 640
    image_width = 640
    nc = 80
    anchors = config.ANCHORS
    x = torch.rand(batch_size, 3, image_height, image_width)
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    pretrained_weights = torch.load("ultralytics_files/yolov5m_state_dict.pt")
    model_arch = torch.load("ultralytics_files/yolov5m.pt")
    pretrained_weights_values = list(pretrained_weights.values())
    num_params = 0

    for idx, (layer, weight) in enumerate(model.state_dict().items()):
        print(weight.shape)

    c=1
