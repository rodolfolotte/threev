import os
import torch
from ultralytics import YOLO

torch.cuda.empty_cache()

coco_train_yaml = os.path.join('/media/rodolfo/data/personal/livecell/data/livecell/dataset.yaml')

model = YOLO('yolov8m-seg.pt')
model.train(data=coco_train_yaml, epochs=10, imgsz=704, device=0, batch=2)