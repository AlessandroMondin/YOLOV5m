import time
from tqdm import tqdm
from model import YOLOV5m
from dataset import MS_COCO_2017_VALIDATION
import config
import torch
from torch.utils.data import DataLoader
from utils.plot_utils import cells_to_bboxes
from utils.bboxes_utils import non_max_suppression as nms, non_max_suppression_aladdin

if __name__ == "__main__":
    check_loss = True
    batch_size = 8
    image_height = 640
    image_width = 640
    nc = 91
    S = [8, 16, 32]

    anchors = config.ANCHORS
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False).to(config.DEVICE)

    dataset = MS_COCO_2017_VALIDATION(num_classes=len(config.COCO_LABELS), anchors=config.ANCHORS,
                                      root_directory=config.ROOT_DIR, transform=config.ADAPTIVE_VAL_TRANSFORM,
                                      train=False, S=S, rect_training=True, default_size=640, bs=8,
                                      coco_128=True)

    anchors = torch.tensor(anchors)

    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

    for i in range(2):
        start = time.time()
        loop = tqdm(loader)
        for img, _ in loop:
            img = torch.stack(img, 0)
            with torch.no_grad():
                out = model(img)
            if i == 0:
                bboxes_1 = cells_to_bboxes(out, anchors, strides=model.head.stride, is_pred=True, list_output=True)
                nms_boxes = []
                for boxes in bboxes_1:
                    nms_boxes.append(non_max_suppression_aladdin(boxes, iou_threshold=0.6, threshold=0.01, max_detections=300))
            if i == 1:
                bboxes = cells_to_bboxes(out, anchors, strides=model.head.stride, is_pred=True, list_output=False)
                bboxes = nms(bboxes, iou_threshold=0.6, threshold=0.01, max_detections=300)
        print(time.time() - start)
        time.sleep(1)



