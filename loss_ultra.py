# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
from utils.bboxes_utils import intersection_over_union


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))


        self.balance = [4.0, 1.0, 0.4]  # explanation.. https://github.com/ultralytics/yolov5/issues/2026
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = model.head.na  # number of anchors
        self.nc = model.head.nc  # number of classes
        self.nl = model.head.nl  # number of layers
        self.anchors = model.head.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = intersection_over_union(pbox, tbox[i], GIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, device=self.device)  # targets
                    t[range(n), tcls[i]] = 1
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h),
        # where "image" is the idx of the image in the batch: i.e. if batch_size is 4,
        # and all the images contain at least 1 target, the value of "image" will be 0,1,2,3
        na, nt = self.na, targets.shape[0]  # number of anchors (x scale), targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # torch.arange(na, device=self.device) i.e --> tensor([0, 1, 2]), shape 3
        # torch.arange(na, device=self.device).float().view(na, 1) --> tensor([[0.], [1.], [2.]]), shape (3, 1)
        # torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt) --> shape (3, 84), in words,
        # ai[0,:] repets for n_detections (here 84) times 0, ai[1,:] repets for n_detections (here 84) times 1, etc.
        # ai.shape --> (na,nt) --> i.e. (3, 84)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # shape of "targets" before this line is (n_gt_detections, 6) where 6 are (image,class,x,y,w,h)

        # targets.repeat(na, 1, 1) has shape (num anchors (x scale ?), n_gt_detections, 6)  i.e. (3, 84, 6)
        # and torch.equal(targets.repeat(na, 1, 1)[0,:], targets.repeat(na, 1, 1)[1,:]) and
        # torch.equal(targets.repeat(na, 1, 1)[0,:], targets.repeat(na, 1, 1)[2,:]) return true

        # ai[..., None]  has shape (na, nt, 1), i.e. (3, 84, 1)

        # torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2) we can the two tensors along the last dim (2) and
        # we get a (na, nt, 7) tensor, i.e. (3, 84, 7)

        g = 0.5  # bias
        # off.shape --> (5,2)
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        # for each detection layer...
        for i in range(self.nl):
            # we get the anchors and the predictions shape of the i-eme detection layer
            anchors, shape = self.anchors[i], p[i].shape
            # anchors shape (na x scale,2), i.e (3,2) -- shape.shape i.e. (8, 3, 80, 80, 85)

            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # we set these values of the gain variable as the third,
            # second, third and second values of the shape matrix i.e. (80, 80, 80, 80)

            # "gain" will now look like this: tensor([ 1.,  1., 80., 80., 80., 80.,  1.])


            # Match targets to anchors
            # targets.shape (3, 82, 7), gain.shape (7), t.shape (3, 82, 7)
            # where the 7 in dim=2 are (image,class,x,y,w,h, detection_layer_idx) and
            # targets * gain multiply x,y,w,h  times w,h,w,h of the detection_layer
            t = targets * gain  # shape(3,n,7)
            # if there are predictions (n predicted bboxes > 0)
            if nt:
                # Matches

                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # t.shape is (na, n_gt_boxes, 7)
                # anchors[:, None].shape is (na x scale, 1, 2), i.e. (3,1,2)
                # so this operation is the ratio between the w, h (that after t = targets * gain are
                # representing w,h in terms of "grid blocks") and anchors

                # r.shape is (na, n_gt_boxes, 2) and represents the ratio explained above (why?)

                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare

                # torch.max(r, 1 / r) has shape (na, n_gt_boxes, 2) and returns the max between
                # each element of the last dimension

                # torch.max(r, 1 / r).max(2) returns torch.return_type.max variable and to
                # access the output tensor you have to subset the first element with torch.max(r, 1 / r).max(2)[0]

                # torch.max(r, 1 / r).max(2)[0] has shape (na, n_gt_boxes) i.e. (3, 84).
                # The last dimension has "disappeared" because .max(2) just picks the maximum
                # element over the last dimension

                # torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t'] has shape (na, n_gt_boxes) i.e. (3, 84)
                # and is a boolean. What is self.hyp['anchor_t']? Check here below:
                # https://github.com/ultralytics/yolov5/issues/1310#issuecomment-723035010

                # So according to Glen, "j are target-anchor match candidates."

                t = t[j]  # filter

                # t.shape (na (xs?), n_gt_detections, 7) i.e. (3, 84, 7)
                # j.shape (na (xs?), n_gt_detections) i.e. (3, 84)

                # t[j] (?!) shape (93, 7), which means that some gt_detections might require
                # more set of anchors because they are borderline???

                # Offsets
                gxy = t[:, 2:4]  # grid xy

                # t[:, 2:4]. shape i.e (93, 2)


                gxi = gain[[2, 3]] - gxy  # inverse
                # gain.shape is 2 i.e. 80, 80 --- gxy shape i.e. (93, 2)
                # gxi.shape == gxy.shape
                # rescale/shift the x, y coordinates, why? and to get what?
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
