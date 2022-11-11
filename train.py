import argparse
import os.path

import torch
from model import YOLOV5m
from loss import YOLO_LOSS
from torch.optim import Adam
from utils.validation_utils import YOLO_EVAL
from utils.training_utils import train_loop, get_loaders
from utils.utils import save_checkpoint, load_model_checkpoint, load_optim_checkpoint
from utils.plot_utils import save_predictions
import config

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco128val", action='store_true', help=" Validate on COCO128")
    parser.add_argument("--nosaveimgs", action='store_true', help="Don't save images predictions in SAVED_IMAGES folder")
    parser.add_argument("--nosavemodel", action='store_true', help="Don't save model weights in SAVED_CHECKPOINT folder")
    parser.add_argument("--epochs", type=int, default=273, help="Num of training epochs")
    parser.add_argument("--nosavelogs", action='store_true', help="Don't save train and eval logs on train_eval_metrics")
    parser.add_argument("--rect", action='store_true', help="Performs rectangular training")
    parser.add_argument("--bs", type=int, default=8, help="Set dataloaders batch_size")
    parser.add_argument("--nw", type=int, default=4, help="Set number of workers")
    parser.add_argument("--resume", action='store_true', help="Resume training on a saved checkpoint")
    parser.add_argument("--filename", type=str, help="Model name to use for resume training")
    parser.add_argument("--load_coco_weights", action='store_true', help="Loads Ultralytics weights, (~273 epochs on MS COCO)")
    parser.add_argument("--only_eval", action='store_true', help="Performs only the evaluation (no training loop")

    return parser.parse_args()


def main(opt):

    first_out = config.FIRST_OUT
    scaler = torch.cuda.amp.GradScaler()

    model = YOLOV5m(first_out=first_out, nc=len(config.COCO80), anchors=config.ANCHORS,
                    ch=(first_out * 4, first_out * 8, first_out * 16), inference=False).to(config.DEVICE)

    optim = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # if no models are saved in checkpoints, creates model0 files,
    # else i.e. if model0.pt is in the folder, new filename will be model1.pt
    starting_epoch = 0
    # if loading pre-existing weights
    if opt.resume:
        filename = opt.filename
        folder = os.listdir(os.path.join("SAVED_CHECKPOINT", opt.filename))
        # getting the epoch of the last filename by str wrangling
        starting_epoch = int(folder[-1].split(".")[0].split("_")[-1]) + 1

        load_model_checkpoint(opt.filename, model)
        load_optim_checkpoint(opt.filename, optim)

    if opt.load_coco_weights:
        model.load_state_dict(torch.load("yolov5_my_arch_ultra_w.pt"), strict=True)
        filename = "ULTRALYTICS_PRETRAINED"

    # elif
    else:
        if "model" not in "".join(os.listdir("SAVED_CHECKPOINT")):
            filename = "model_1"
        else:
            last_model_name = os.listdir("SAVED_CHECKPOINT")[-1]
            filename = "model_" + str(int(last_model_name.split("_")[1]) + 1)

    save_logs = False if opt.nosavelogs else True
    rect_training = True if opt.rect else False

    # check get_loaders to see how augmentation is set
    train_loader, val_loader = get_loaders(db_root_dir=config.ROOT_DIR, batch_size=opt.bs,
                                           num_workers=opt.nw, rect_training=rect_training,
                                           coco128val=opt.coco128val)

    loss_fn = YOLO_LOSS(model, save_logs=save_logs, rect_training=rect_training,
                        filename=filename, resume=opt.resume)

    evaluate = YOLO_EVAL(save_logs=save_logs, conf_threshold=config.CONF_THRESHOLD,
                         nms_iou_thresh=config.NMS_IOU_THRESH,  map_iou_thresh=config.MAP_IOU_THRESH,
                         device=config.DEVICE, filename=filename, resume=opt.resume)

    # starting epoch is used only when training is resumed by loading weights
    for epoch in range(0 + starting_epoch, opt.epochs + starting_epoch):

        model.train()

        if not opt.only_eval:
            train_loop(model=model, loader=train_loader, loss_fn=loss_fn, optim=optim,
                       scaler=scaler, epoch=0+starting_epoch, num_epochs=opt.epochs + starting_epoch,
                       multi_scale_training=not rect_training)

        model.eval()

        evaluate.check_class_accuracy(model, val_loader)

        evaluate.map_pr_rec(model, val_loader, anchors=model.head.anchors, epoch=epoch+1)

        # NMS WRONGLY MODIFIED TO TEST THIS FEATURE!!
        if not opt.nosaveimgs:
            save_predictions(model=model, loader=val_loader, epoch=epoch, num_images=5,
                             folder="SAVED_IMAGES", device=config.DEVICE, filename=filename)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        if not opt.nosavemodel:
            save_checkpoint(checkpoint, folder_path="SAVED_CHECKPOINT", filename=filename, epoch=epoch+1)


if __name__ == "__main__":
    parser = arg_parser()
    main(parser)

