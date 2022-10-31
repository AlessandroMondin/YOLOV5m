import json
import os

from tqdm import tqdm
import pprint
import csv
from time import sleep
"""
with open("/Users/Alessandro/desktop/ML/DL_DATASETS/COCO/annotations/instances_train2017.json") as f:
    coco_ann = json.load(f)

coco_parsed = []

loop = tqdm(coco_ann["images"])

for idx, image in enumerate(loop):
    if idx < 16:
        name = image["file_name"]
        h, w = image["height"], image["width"]
        id = image["id"]
        bboxes = []

        for ann in coco_ann["annotations"]:
            if ann["image_id"] == id:
                box = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
                bboxes.append({"box": box, "class": ann["category_id"]})

        coco_parsed.append(
            {"img_name": name,
             "height": h,
             "width": w,
             "bboxes": bboxes}
        )
    else:
        break


path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_16.json"

with open(path, "w") as f:
    json.dump(coco_parsed, f)


path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_2017_val_AM.json"
with open(path, "r") as f:
    annotations = json.load(f)

loop = tqdm(annotations)

for annot in loop:
    img_name = annot["img_name"]
    height = annot["height"]
    width = annot["width"]

    with open("/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/"
              "annotations/coco_2017_val_csv.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([img_name, height, width])
        f.close()

    folder_path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_2017_val_txt"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path,"{}.txt".format(img_name[:-4])), "w") as fp:

        boxes = [box["box"]+[box["class"]] for box in annot["bboxes"]]
        for bbox in annot["bboxes"]:
            box = bbox["box"]
            x, y, w, h = box
            # cleans empty bboxes
            if w > 0.1 and h > 0.1:
                w = width if w >= width else w
                h = height if h >= height else h
                box = [x, y, w, h, bbox["class"]]
                fp.write(str(box).strip("[]").replace(",","") + "\n")
        fp.close()
"""

# widths = []
# heights = []

# here the intuition:
# in the YOLOV5 repo they used nearest interpolation also to increase the size of images.
# IMO this could lead to some distortions, hence I am trying to use interpolation just
# to randomly decrease their value if they are large enough (in my case
# w.r.t. the 75% percentile)

# print("width mean before multi_shape: {:.2f}".format(sum(widths)/len(widths)))
# print("height mean before multi_shape: {:.2f}".format(sum(heights)/len(heights)))
# percentile_75_w = np.percentile(widths, 75)
# percentile_75_h = np.percentile(heights, 75)

# wh = []
# for ann in annot:
#    t = (ann["height"], ann["width"])
#    h, w = my_interpolation(t)
#    wh.append((h, w))

# widths = [i[1] for i in wh]
# heights = [i[0] for i in wh]

# print("width mean post multi_shape: {:.2f}".format(sum(widths)/len(widths)))
# print("height mean post multi_shape: {:.2f}".format(sum(heights)/len(heights)))
# print(Counter(wh))