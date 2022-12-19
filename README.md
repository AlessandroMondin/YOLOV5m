## A personal implementation of YOLOv5(m)

This repository is thought to facilitate the understanding of YOLOv5 by creating a simple file structure focused on the PyTorch implementation.
The sources I used while implementating are Ultralytics <a href="https://github.com/ultralytics/yolov5" target="_blank">YOLOv5</a> for architecture details and Aladdin Persson <a href="https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3" target="_blank">YOLOv3</a> for the training-pipeline design.<br> 

## Instruction for use:

    git clone https://github.com/AlessandroMondin/YOLOV5m.git
    cd YOLOv5m
    pip install -r requirements.txt

<i>I am updating it soon with a bried guide on how to use it.</i> <br> <br> 

After loading Ultralytics COCO weights in my architecture, I've fine-tuned it on <a href="https://universe.roboflow.com/thermal-imaging-0hwfw/flir-data-set/dataset/14">FLIR dataset<a/> for ~15 epoch and it reached ~0.82 i MAP50


<img src="https://github.com/AlessandroMondin/computer_vision/blob/main/yolov5/doc_files/yolo_v5_architecture.png" 
     width="800" 
     height="800"
     class="center"/>


