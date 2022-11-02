import time
import os
import csv
import torch
from tqdm import tqdm
from utils.bboxes_utils import intersection_over_union, non_max_suppression, my_nms
from utils.plot_utils import cells_to_bboxes
from collections import Counter


# My class that integrates ALADDIN's functions

class YOLO_EVAL:
    def __init__(self, save_logs, conf_threshold, nms_iou_thresh, map_iou_thresh, device, filename, resume):

        self.save_logs = save_logs

        self.conf_threshold = conf_threshold  # considers for nms only bboxes with obj_score > conf_thres
        self.nms_iou_thresh = nms_iou_thresh  # during nms keeps all same class bboxes with iou < nms_iou_thresh w.r.t. each others
        self.map_iou_threshold = map_iou_thresh  # while computing map, a prediction to be classified as true positive needs to have iou > map_theshold w.r.t. gt bbox

        self.device = device
        self.filename = filename

        if self.save_logs:
            if not resume:
                folder = os.path.join("train_eval_metrics", filename)
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                with open(os.path.join(folder, "eval.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "class_accuracy",
                                     "noobj_accuracy", "obj_accuracy",
                                     "mapval", "precision", "recall"])
                    print("--------------------------------------------------------------------------------------")
                    print(f'Eval Logs will be saved in {os.path.join("train_eval_metrics", filename, "eval.csv")}')
                    print("--------------------------------------------------------------------------------------")
                    f.close()

        # map and accuracies are computed in 2 distinct methods, to use csv.writerow and store
        # map metrics and accuracies all at once the three below are set as class attributes
        # initialising them here for readability
        self.class_accuracy = None
        self.noobj_accuracy = None
        self.obj_accuracy = None

    def check_class_accuracy(self, model, loader):
        model.eval()
        print(".. Computing: class, no-obj and obj accuracies ..")
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for idx, (images, y) in enumerate(tqdm(loader)):

            images = images.to(self.device)
            with torch.no_grad():
                out = model(images)

            for i in range(3):
                y[i] = y[i].to(self.device)
                obj = y[i][..., 0] == 1  # in paper this is Iobj_i
                noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

                correct_class += torch.sum(
                    torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
                )
                tot_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > self.conf_threshold
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                tot_obj += torch.sum(obj)
                correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                tot_noobj += torch.sum(noobj)

        time.sleep(1) # to not broke tqdm logger

        class_accuracy = (correct_class / (tot_class_preds + 1e-16))
        noobj_accureacy = (correct_noobj / (tot_noobj + 1e-16))
        obj_accuracy = (correct_obj / (tot_obj + 1e-16))

        if self.save_logs:
            self.class_accuracy = round(float(class_accuracy), 3)
            self.noobj_accuracy = round(float(noobj_accureacy), 3)
            self.obj_accuracy = round(float(obj_accuracy), 3)

        print("Class accuracy: {:.2f}%".format(class_accuracy * 100))
        print("No obj accuracy: {:.2f}%".format(noobj_accureacy * 100))
        print("Obj accuracy: {:.2f}%".format(obj_accuracy * 100))

        model.train()

    def map_pr_rec(self, model, loader, anchors, num_classes, epoch):

        print(".. Computing: MAP, Precision and Recall ..")
        # make sure model is in eval before get bboxes

        model.eval()

        # 1) GET EVALUATION BBOXES
        train_idx = 0
        pred_boxes = []
        true_boxes = []
        for batch_idx, (x, labels) in enumerate(tqdm(loader)):
            x = x.to(self.device)
            with torch.no_grad():
                predictions = model(x)

            batch_size = x.shape[0]

            bboxes = cells_to_bboxes(predictions, anchors, strides=model.head.stride, is_pred=True)

            # we just want one bbox for each label, not one for each scale
            true_bboxes = cells_to_bboxes(labels, anchors, strides=model.head.stride, is_pred=False)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=self.nms_iou_thresh,
                    threshold=self.conf_threshold,
                    box_format="midpoint",
                )

                for box in nms_boxes:
                    pred_boxes.append([train_idx] + box)

                for box in true_bboxes[idx]:
                    if box[1] > self.conf_threshold:
                        true_boxes.append([train_idx] + box)

                # it increases by one after each image (not after each batch!
                train_idx += 1

        # 2) COMPUTE MAP, PRECISION AND RECALL

        """
        Video explanation of this function:
        https://youtu.be/FppOzcDvaDI
        This function calculates mean average precision (mAP)
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes
        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """

        # list storing all AP for respective classes
        average_precisions = []
        all_precisions = []
        all_recalls = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            # number of gtbboxes for each val image
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by objectness score which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0 and len(detections) == 0:
                continue
            # I added this condition: if num gts is 0 but len detections is not, all
            # detections are going to be false positives
            elif total_true_bboxes == 0 and len(detections) != 0:
                FP = torch.ones((len(detections)))
            else:
                for detection_idx, detection in enumerate(detections):
                    # Only take out the ground_truths that have the same
                    # training idx as detection
                    ground_truth_img = [
                        bbox for bbox in ground_truths if bbox[0] == detection[0]
                    ]

                    best_iou = 0

                    for idx, gt in enumerate(ground_truth_img):
                        iou = intersection_over_union(
                            torch.tensor(detection[3:]),
                            torch.tensor(gt[3:]),
                            box_format="midpoint",
                        )

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou > self.map_iou_threshold:
                        # only detect ground truth detection once
                        if amount_bboxes[detection[0]][best_gt_idx] == 0:
                            # true positive and add this bounding box to seen
                            TP[detection_idx] = 1
                            amount_bboxes[detection[0]][best_gt_idx] = 1
                        else:
                            FP[detection_idx] = 1
                    # if IOU is lower then the detection is a false positive
                    else:
                        FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))  # why?? Numerical stability?
            recalls = torch.cat((torch.tensor([0]), recalls))  # why?? Numerical stability?
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))
            all_precisions.append(sum(precisions.tolist())/len(precisions.tolist()))
            all_recalls.append(sum(recalls.tolist())/len(recalls.tolist()))

        mapval = round((sum(average_precisions) / len(average_precisions)).item(), 3)
        avg_precision = round(sum(all_precisions) / len(all_precisions), 3)
        avg_recall = round(sum(all_recalls) / len(all_recalls), 3)

        time.sleep(1)  # to not broke tqdm logger
        print(f"MAP: {mapval}, \nPrecision: {avg_precision}, \nRecall: {avg_recall}")

        if self.save_logs:
            with open(os.path.join("train_eval_metrics", self.filename, "eval.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, self.class_accuracy, self.noobj_accuracy,
                                 self.obj_accuracy, mapval, avg_precision, avg_recall])

        model.train()

        return mapval, avg_precision, avg_recall
