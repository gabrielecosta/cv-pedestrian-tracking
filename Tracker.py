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

# Funzione per plottare i risultati
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

# Funzione per riscalare le bounding boxes
def rescale_bboxes(boxes, size):
    
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(boxes)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    return b

def plot_image_w_detections(image, detections):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for detection in detections:
        frame,id,x,y,w,h,conf,_,_,_ = detection
        rectangle = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rectangle)

        # Step 4: Add text
        # Define the text and its position
        text = f"id: {id}, conf:{conf:.2f}"
        text_position = (x, y-10)  # Position the text at the top-left corner with some padding
        # Add the text to the plot with alignment properties
        ax.text(*text_position, text, fontsize=5, color='green',
        verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))


    # Step 5: Display the image
    plt.axis('off')  # Turn off the axis
    plt.show()

class sim_VGG16_net:
    def __init__(self):
        
        # Carica il modello VGG16 pre-addestrato
        self.base_model = VGG16(weights='imagenet', include_top=True)
        
        # Estrai l'output dello strato prima dell'ultimo strato completamente connesso
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('fc2').output)

    # Funzione per caricare e pre-processare un'immagine
    def load_and_preprocess_image(self, frame):
        
        # img = image.load_img(image_path, target_size=(224, 224))
        # VGG accetta in input immagini 224x224
        frame = frame.convert('RGB')
        img_resized = frame.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    # Funzione per estrarre le features da un'immagine
    def extract_features_nb(self, frame):
        
        img = self.load_and_preprocess_image(frame)
        features = self.model.predict(img)
        return features.flatten()

    # Funzione per estrarre le features da un'immagine
    def extract_features(self, frame, bbox):
        
        # Definisci il bounding box per il crop
        # (left, upper): The coordinates of the top-left corner of the bounding box.
        # (right, lower): The coordinates of the bottom-right corner of the bounding box.
        x1,y1,x2,y2 = bbox  # (left, upper, right, lower)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        sub_box = (x1,y1,x2,y2)
        subbox = frame.crop(sub_box)
        img = self.load_and_preprocess_image(subbox)
        
        # per togliere il verbose model.predict(x,verbose=0)
        # features = self.model.predict(img)
        features = self.model.predict(img,verbose=0)
        return features.flatten()

    # Funzione per calcolare la similarità tra due immagini basata sulla distanza coseno delle features
    def calculate_similarity(self, frame1, frame2):
        features1 = self.extract_features(frame1)
        features2 = self.extract_features(frame2)
        # Calcola la distanza coseno tra le features
        similarity = 1 - cosine(features1, features2)
        return similarity

    def calculate_similarity_reid(self, frame1, features2):
        features1 = self.extract_features(frame1)
        similarity = 1 - cosine(features1, features2)
        return similarity

    def calulate_similarity_features(self,features1, features2):
        similarity = 1 - cosine(features1, features2)
        return similarity

# Istanziamento di una VGG16
vgg16 = sim_VGG16_net()

class Tracker:

    def __init__(self):

        # Lista di tracker
        self.trackers = []

        # Contatore per assegnare ID univoci ai pedoni
        self.track_counter = 0

        # Quanti frame devo aspettare prima che il tracker venga rimosso dall'immagine
        self.max_lost_frames = 30

        # Si definisce un vettore di vanishing tracks, il quale serve per determinare i track precedentemente persi.
        
        # Rispecchia la struttura di un tracker, ma è formato da <id>, <feature_desc> e <lost>
        self.vanishing_tracks = []

        # Si definisce un vettore di track morti
        self.dead_tracks = []

    def update_tracker(self, confidences, detections, frame, vgg16, threshold_det_track=1.0, threshold_reid=1.0):

        # Se non ci sono tracker, significa che è il primo insieme di rilevamenti, quindi bisogna aggiungere ogni nuovo oggetto tracciato
        if not self.trackers:
            
            # Aggiunta delle detection rilevate alla lista di tracker
            for (detection, conf) in zip(detections, confidences):

                track = {"bbox": detection, "id" : self.track_counter, "conf": conf, "lost": 0}
                self.trackers.append(track)
                self.track_counter += 1

        # In caso contrario, bisogna gestire i rilevamenti esistenti
        else:

            # Memorizza i rilevamenti esistenti, ovvero le identità presenti al momento
            current_bboxes = [tracker['bbox'] for tracker in self.trackers if tracker['lost'] == 0]
            current_frames = [tracker for tracker in self.trackers if tracker['lost'] == 0]

            # Calcolo della matrice di costo
            cost_matrix = np.array([[self.compute_cost(tracker, detection) for detection in detections] for tracker in current_bboxes])
            if cost_matrix.max() != 0:
            # Normalizzazione della matrice di costo
                norm_cost_matrix = cost_matrix / cost_matrix.max()
            else:
                norm_cost_matrix = 0
            # Applicazione dell'algoritmo per il bipartite matching (algoritmo ungherese)
            row_indices, col_indices = linear_sum_assignment(norm_cost_matrix)

            # Crea una lista di coppie con le corrispondenze ottimali
            matched_indices = list(zip(row_indices, col_indices))

            # Crea dei set di detection e tracker non matchati
            unmatched_detections = set(range(len(detections))) - set(col_indices)
            unmatched_trackers = set(range(len(current_bboxes))) - set(row_indices)


            # Iterazione su tutte le coppie di indici corrispondenti ottenute dall'algoritmo ungherese.
            # Per ogni coppia aggiorna il bounding box e rimposta il contatore di fotogrammi persi a zero perché il rilevamento dell'oggetto continua.
            
            for t_idx, d_idx in matched_indices:
                
                # qui bisogna mettere una soglia sulle assegnazioni corrispondenti,
                # se il valore della matrice di costo C[t_idx, d_idx] è maggiore di un certo valore allora assegna
                # altrimenti è un lost!
                # usare theshold_det_track
                # print(norm_cost_matrix[t_idx, d_idx])
                # se stanno sotto la soglia allora vanno bene, altrimenti devo scartarli
                # sotto perché è un problema di minimo
                
                id_track = current_frames[t_idx]['id']
                
                if self.trackers[id_track]['lost'] == 0:
                    
                    # questo vale solo per le non lost detections
                    if norm_cost_matrix[t_idx, d_idx] <= threshold_det_track:
                        print(f"Tracker con {id_track} identificato tra i frame.")
                        self.trackers[id_track]['bbox'] = detections[d_idx] # aggiorna la bounding box
                        self.trackers[id_track]['conf'] = confidences[d_idx] # aggiorna la confidence
                        self.trackers[id_track]['lost'] = 0 # aggiorna il numero di frame persi

                    else:
                        
                        self.trackers[id_track]['lost'] += 1
                        
                        bbx = self.trackers[id_track]['bbox']
                        
                        feature_lost = vgg16.extract_features(frame,bbx)
                        
                        lost_track = {'id':id_track, 'bbox': self.trackers[id_track]['bbox'], 'conf': self.trackers[id_track]['conf'], 'feature': feature_lost}
                        
                        self.vanishing_tracks.append(lost_track)

            # Aggiungi nuovi rilevamenti che non hanno corrispondenze precedenti alla lista dei tracker
            remaining_detection = [detections[d_idx] for d_idx in unmatched_detections]
            remaining_confidences = [confidences[d_idx] for d_idx in unmatched_detections]

            if len(self.vanishing_tracks) != 0 and len(unmatched_detections) != 0:

                vanishing_features = [vanishing['feature'] for vanishing in self.vanishing_tracks]
                remaining_features = [vgg16.extract_features(frame, det) for det in remaining_detection]
                sim_matrix = np.array([[vgg16.calulate_similarity_features(vanishing_feature, remaining_feature) for remaining_feature in remaining_features ] for vanishing_feature in vanishing_features])
                IoU_matrix = np.array([[self.iou(van['bbox'], det) for det in remaining_detection] for van in self.vanishing_tracks])
                norm_cost_matrix_reid = 0.6 * (1 - sim_matrix) + 0.4 * (1 - IoU_matrix)

                row_indices, col_indices = linear_sum_assignment(norm_cost_matrix_reid)

                # Crea una lista di coppie con le corrispondenze ottimali
                matched_indices_reid = list(zip(row_indices, col_indices)) # ho matchato le vanishing

                vanishing_list_enumerate = []
                
                for t_idx, det in enumerate(self.vanishing_tracks):
                    vanishing_list_enumerate.append(det)

                # riassegno le matched solo se hanno un valore di soglia opportuno
                for t_idx, d_idx in matched_indices_reid:
                    
                    feature_1_test = vgg16.extract_features(frame,remaining_detection[d_idx])
                    feature_2_test = vanishing_list_enumerate[t_idx]['feature']
                    
                    print(f'Similarità: {vgg16.calulate_similarity_features(feature_1_test,feature_2_test)}')
                    
                    if norm_cost_matrix_reid[t_idx, d_idx] <= threshold_reid:
                        
                        # aggiorno utilizzando l'ID delle vanished che cammina di pari passo con trackers
                        # con le remaining_detections
                        print(f"Effettuata reidentificazione: {vanishing_list_enumerate[t_idx]['id']}")
                        self.trackers[vanishing_list_enumerate[t_idx]['id']]['bbox'] = remaining_detection[d_idx] # aggiorna la bounding box
                        self.trackers[vanishing_list_enumerate[t_idx]['id']]['conf'] = remaining_confidences[d_idx] # aggiorna la confidence
                        self.trackers[vanishing_list_enumerate[t_idx]['id']]['lost'] = 0 # aggiorna il numero di frame persi
                        # tolgo da vanishing
                        self.vanishing_tracks = [t for t in self.vanishing_tracks if t['id'] != vanishing_list_enumerate[t_idx]['id']]
                    
                    else:
                        
                        print(f"Valore sopra la soglia per la re-id: {norm_cost_matrix_reid[t_idx, d_idx]}")
                        
                        # se non reidentifico allora devo creare una nuova detection
                        new_track = {'bbox': remaining_detection[d_idx], 'id': self.track_counter, 'conf': remaining_confidences[d_idx], 'lost':0}
                        self.trackers.append(new_track)
                        self.track_counter += 1

                # Crea dei set di detections non matchate con le vanishing, quindi nuove detections
                unmatched_unmatched_detections = set(range(len(remaining_detection))) - set(col_indices)

                # adesso provo a matchare e unmatchare con i vanishing
                # se matchano allora provvedo a reinserire nel tracker l'id a lost=0,
                # aggiorno la confidence e la boundary box sulla base della matchata
                # altrimenti la devo assegnae nuova
                
                for d_idx in unmatched_unmatched_detections:
                    print("New detection after re-id not found!")
                    new_track = {'bbox': remaining_detection[d_idx], 'id': self.track_counter, 'conf': remaining_confidences[d_idx], 'lost':0}
                    self.trackers.append(new_track)
                    self.track_counter += 1
            else:
                # ancora non ci sono track scomparse
                for d_idx in unmatched_detections:
                    
                    print("New detection!")
                    new_track = {'bbox': detections[d_idx], 'id': self.track_counter, 'conf':confidences[d_idx], 'lost':0}
                    self.trackers.append(new_track)
                    self.track_counter += 1

            vanished_keys = [vanished['id'] for vanished in self.vanishing_tracks]

            # devo aggiornare i track persi di quelli già persi
            for lost_vanished_key in vanished_keys:
                if self.trackers[lost_vanished_key]['lost'] > 0:
                    self.trackers[lost_vanished_key]['lost'] += 1

            # Aumenta il contatore per i tracker persi al frame corrente!
            # stampo gli unmatched track
            for t_idx in unmatched_trackers:

                id_track = current_frames[t_idx]['id']
                print(f"Lost track id: {id_track}")
                self.trackers[id_track]['lost'] += 1
                bbx = self.trackers[id_track]['bbox']
                print(f'Track già scomparsa? {id_track not in vanished_keys}')
                print(f'Track scomparse: {self.vanishing_tracks}')
                if id_track not in vanished_keys:
                    feature_lost = vgg16.extract_features(frame,bbx)
                    lost_track = {'id':id_track, 'bbox': self.trackers[id_track]['bbox'], 'conf': self.trackers[id_track]['conf'], 'feature': feature_lost}
                    self.vanishing_tracks.append(lost_track)


            keep_tracks = []
            
            # le riaggiorno
            vanished_keys = [vanished['id'] for vanished in self.vanishing_tracks]

            for lost_vanished_key in vanished_keys:
                
                if self.trackers[lost_vanished_key]['lost'] > self.max_lost_frames:
                    self.dead_tracks.append(self.trackers[lost_vanished_key]['id'])
                    id_dead = self.trackers[lost_vanished_key]['id']
                    print(f'Morta la track {id_dead}')
                
                if self.trackers[lost_vanished_key]['lost'] > 0 and self.trackers[lost_vanished_key]['lost'] <= self.max_lost_frames:
                    keep_tracks.append(self.trackers[lost_vanished_key]['id'])

            # tolgo da vanishing le track perse ma non ancora morte
            self.vanishing_tracks = [t for t in self.vanishing_tracks if t['id'] in keep_tracks]
            print(f'Track perse conservate: {self.vanishing_tracks}')


    def compute_cost(self, tracker, detection):
        # t_x1, t_y1, t_x2, t_y2 = tracker
        # d_x1, d_y1, d_x2, d_y2 = detection
        # dist = np.linalg.norm(np.array([(t_x1+t_x2)/2, (t_y1+t_y2)/2]) - np.array([(d_x1+d_x2)/2, (d_y1+d_y2)/2]))
        iou = self.iou(tracker, detection)
        return (1-iou) # la distanza varia tra 0 e 1

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

# Sequence file
sequence_file = "C:/Users/caste/OneDrive/Desktop/Assignment 3/dataset/NEWMOT17-11-DPM-DETR07.txt"

data_preloaded = []

with open(sequence_file, "r") as f:

    for row in f:

        frame, x1, x2, y1, y2, confidence = row.split(",")

        frame = int(frame)
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        conf = float(confidence)

        detection_frame = [frame, [x1, x2, y1, y2], conf]
        data_preloaded.append(detection_frame)

final_detections = {}
for data in data_preloaded:
    if data[0] not in final_detections.keys():
        final_detections[data[0]] = []
    final_detections[data[0]].append([data[1], data[2]])

detections_frame_1 = final_detections[1]
detections_frame_1

for detection, confidence in final_detections[1]:
    print(f'Detection: {detection}; Confidence: {confidence}')

def process_image_folder_preloaded(detections_loaded, folder_path, frame_size=(640, 360), detection_interval=1, frame_limit_flag = False, limit=5, threshold_det_track=0.4, threshold_reid=0.4):
    tracker = Tracker()
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frame_count = 0

    detections_frame = []

    for frame_file in frame_files:
        # detr_preloaded = []

        if frame_limit_flag and frame_count > limit:
            break
        frame = Image.open(frame_file)

        if frame_count % detection_interval == 0:
            # confidences, detections = detect_pedestrians(im=frame, model=model)
            detr_preloaded = detections_loaded[frame_count]
        detections_detr = []
        confidences_detr = []
        for detr in detr_preloaded:
            detections_detr.append(detr[0])
            confidences_detr.append(detr[1])
        # print(detections_detr)
        # quindi da qua in poi non cambia più nulla rispetto a prima
        tracker.update_tracker(confidences_detr, detections_detr, frame, vgg16, threshold_det_track, threshold_reid)

        actual_detections = [] # solo per print

        for track in tracker.trackers:
            if track['lost'] == 0:
                x1, y1, x2, y2 = map(int, track['bbox'])
                x = x1
                y = y1
                w = x2-x1
                h = y2-y1
                conf = track['conf']
                # poi format_detection deve essere stampato in un file
                format_detectetion = [frame_count, track['id'], x,y,w,h, track['conf'],-1,-1,-1]
                print(format_detectetion)
                actual_detections.append(format_detectetion) # solo per printing
                detections_frame.append(format_detectetion)
        print(f'Frame: {frame_count}')
        frame_count += 1
    return detections_frame

# Images path
images_path = f"C:/Users/caste/OneDrive/Desktop/Assignment 3/dataset/train/MOT17-11-DPM/img1"

# soglia matching = 0.4, soglia re_id = 0.4
detections_to_save_0404 = process_image_folder_preloaded(final_detections, images_path, frame_size=(1920, 1080), detection_interval=1, frame_limit_flag=False, limit=10, threshold_det_track=0.5, threshold_reid=0.3)

# Save path
save_path = f"C:/Users/caste/Unipa - IntelliCrafters/Visione artificiale/Assignments/Assignment 3/outputs/NEWMOT17-11-DPM-06-05-03.txt"

with open(save_path, 'w') as f:
    
    for d in detections_to_save_0404:
        
        frame,id,x,y,w,h,conf,_,_,_ = d
        
        print(f'{frame+1},{id},{x},{y},{w},{h},{conf},-1,-1,-1', file=f)
        
       

images_folder = images_path
detections_file = save_path
output_video_path = f"C:/Users/caste/Unipa - IntelliCrafters/Visione artificiale/Assignments/Assignment 3/outputs/NEWMOT17-11-DPM-06-05-03.mp4"

def draw_video(detections_file, images_folder, output_video_path):
    # Lettura delle detezioni dal file
    detections = {}
    with open(detections_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])  # Numero del frame
            person_id = int(parts[1])  # ID della persona
            x = int(parts[2])  # Coordinata x della bounding box
            y = int(parts[3])  # Coordinata y della bounding box
            w = int(parts[4])  # Larghezza della bounding box
            h = int(parts[5])  # Altezza della bounding box
            
            if frame not in detections:
                detections[frame] = []
            detections[frame].append((x, y, w, h, person_id))

    # Lettura delle immagini e annotazione del video
    image_files = sorted([img for img in os.listdir(images_folder) if img.endswith('.jpg')])
    
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Lettura della prima immagine per ottenere le dimensioni
    first_image_path = os.path.join(images_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Definizione del video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        frame_number = int(img_file.split('.')[0])  # Estrazione del numero del frame dal nome del file
        image = cv2.imread(img_path)

        if frame_number in detections:
            for (x, y, w, h, person_id) in detections[frame_number]:
                # Disegna la bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Annotazione dell'ID della persona
                cv2.putText(image, f'ID: {person_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Aggiunge il frame annotato al video
        video.write(image)

    video.release()
    cv2.destroyAllWindows()

draw_video(save_path, images_path,output_video_path)