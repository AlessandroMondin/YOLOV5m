
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'# or file, Path, PIL, OpenCV, numpy, list
#img = torch.rand(2, 3, 640, 640)

# Inference
results = model(img)

# Results
results.print()
results.show()