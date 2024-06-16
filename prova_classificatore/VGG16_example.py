import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

# Carica il modello VGG16 pre-addestrato
base_model = VGG16(weights='imagenet', include_top=True)
# Estrai l'output dello strato prima dell'ultimo strato completamente connesso
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Funzione per caricare e pre-processare un'immagine
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Funzione per estrarre le features da un'immagine
def extract_features(image_path, model):
    img = load_and_preprocess_image(image_path)
    features = model.predict(img)
    return features.flatten()

# Funzione per calcolare la similarità tra due immagini basata sulla distanza coseno delle features
def calculate_similarity(image1_path, image2_path, model):
    features1 = extract_features(image1_path, model)
    print(features1.shape)
    features2 = extract_features(image2_path, model)
    # Calcola la distanza coseno tra le features
    similarity = 1 - cosine(features1, features2)
    return similarity

# Percorsi delle due immagini da confrontare
image1_path = 'image_test_1.jpg'
image2_path = 'sliced_image1.jpg'

# Calcola la similarità tra le due immagini
similarity = calculate_similarity(image1_path, image2_path, model)
print("Similarity between the two images:", similarity)
