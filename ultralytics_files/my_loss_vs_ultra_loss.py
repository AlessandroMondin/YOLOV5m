import config
import torch
from model import YOLOV5m
from loss import YOLO_LOSS
from ultralytics_loss import ComputeLoss

check_loss = True
batch_size = 8
image_height = 640
image_width = 640
nc = len(config.COCO80)
S = [8, 16, 32]

anchors = config.ANCHORS
first_out = 48

model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                ch=(first_out * 4, first_out * 8, first_out * 16), inference=False).to(config.DEVICE)

#model.load_state_dict(state_dict=torch.load("yolov5m_coco.pt"), strict=True)

ultra_loss = ComputeLoss(model, save_logs=False, filename="none")
my_loss = YOLO_LOSS(model, rect_training=True)

if __name__ == "__main__":
    torch.manual_seed(355)

    # ULTRA_LOSS
    images = torch.rand((4, 3, 640, 640))
    img_idx = torch.arange(4).repeat(3,1).T.reshape(12,1)
    classes = torch.arange(4).repeat(3,1).T.reshape(12,1)
    bboxes = torch.randint(low=0, high=50, size=(12,4))/100
    labels = torch.cat([img_idx, classes, bboxes], dim=-1)
    print(ultra_loss(model(images), labels))

    # MY_LOSS
    images = torch.rand((4, 3, 640, 640))
    bboxes = torch.randint(low=0, high=50, size=(12,4))/100
    labels = torch.cat([bboxes, classes], dim=-1).reshape(4,3,-1).tolist()

    print(my_loss(model(images), labels, pred_size=images.shape[2:4]))










