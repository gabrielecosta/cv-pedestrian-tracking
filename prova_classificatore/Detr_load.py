import  torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

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
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def extract_features(model, im, transform=None):
    if transform is None:
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    # Estrai le features dall'immagine utilizzando il backbone del modello
    # Esegui un'inferenza completa utilizzando il modello
    outputs = model(img)
    # Estrai le features dall'uscita dell'encoder
    features = outputs['encoder_out']
    # Restituisci le features estratte
    return features

def filtered_detect(model, im, transform=None, threshold_confidence=0.7, target_class='person'):
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

    # propagate through the model
    outputs = model(img)

    # keep only predictions with a confidence > threshold_confidence and belonging to the target class
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = (probas.max(-1).values > threshold_confidence) & (probas.argmax(-1) == CLASSES.index(target_class))

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled  


model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
model.load_state_dict(torch.load('detr_model_weights.pth'))

image_path ="im.jpg"
img = Image.open(image_path)
prob, bboxes_scaled = filtered_detect(model, img)

bboxes = []

for p, (xmin, ymin, xmax, ymax), c in zip(prob, bboxes_scaled.tolist(), COLORS * 100):
    bboxes.append([xmin,ymin,xmax-xmin,ymax-ymin]) # formato x,y,w,h

print(bboxes)

# Converte l'immagine in un array NumPy
img_np = np.array(img)

i = 0

for (x,y,w,h) in bboxes:
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
# Esempio di slicing: seleziona solo una parte dell'immagine (ad esempio, i primi 100x100 pixel)
    sliced_img_np = img_np[y:y+h, x:x+w]
    sliced_img_pil = Image.fromarray(sliced_img_np)
    # Salva la parte selezionata dell'immagine
    sliced_img_pil.save("sliced_image" + str(i) + ".jpg")
    i += 1



# features = extract_features(model, img)
# print(prob)
# plot_results(img, prob, bboxes_scaled)




























