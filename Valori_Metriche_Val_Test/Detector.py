# Importazione delle librerie necessarie
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

# Caricamento del modello RESNET50
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Funzione per convertire x, y, w e h in (x1, y1) e (x2, y2)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=1)

# Funzione per riscalare le bounding boxes
def rescale_bboxes(boxes, size):
    
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(boxes)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    return b

# Funzione per rilevare i pedoni (modifica di quella della prof)
def detect_pedestrians(threshold_confidence, model, im, transform = None):
    if transform is None:

        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    outputs = model(img)

    # keep only predictions with a confidence > threshold_confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    max_probas = probas.max(-1).values
    keep = probas.max(-1).values > threshold_confidence
    labels = probas.argmax(-1)

    # Filter by pedestrian
    keep = keep & (labels == 1)

    # Extract the confidences for the kept boxes
    confidences = max_probas[keep].detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return confidences, bboxes_scaled.tolist()

def extract_detections(folder_path, t):
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frame_count = 0

    # preload frames
    for frame_file in frame_files:
        frame = Image.open(frame_file)
        print(f'Frame: {frame_count}')
        confidences, detections = detect_pedestrians(t, im=frame, model=model)

        detection_per_frame = []
        for i in range(len(detections)):
            detection_per_frame.append([detections[i], confidences[i]])

        with open("C:/Users/caste/OneDrive/Desktop/Assignment 3/dataset/NEWMOT17-11-DPM-DETR07.txt", "a") as f:
            for i in range(len(detection_per_frame)):

                detections_str = ", ".join([str(value) for value in detection_per_frame[i][0]])

                print(f'{frame_count}, {detections_str}, {detection_per_frame[i][1]}', file=f)
            
        frame_count += 1



path = "C:/Users/caste/OneDrive/Desktop/Assignment 3/dataset/train/MOT17-11-DPM/img1"
extract_detections(path, t=0.7)

