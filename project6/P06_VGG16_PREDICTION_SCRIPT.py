import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# import du modèle
MODEL_PATH = "model_VGG16_20_0.005.hdf5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# import des classes
BREEDS = {
    0: 'n02106166-Border_collie',
    1: 'n02089078-black-and-tan_coonhound',
    2: 'n02086079-Pekinese',
    3: 'n02108915-French_bulldog',
    4: 'n02094258-Norwich_terrier',
    5: 'n02093754-Border_terrier',
    6: 'n02093256-Staffordshire_bullterrier',
    7: 'n02090379-redbone',
    8: 'n02088632-bluetick',
    9: 'n02107312-miniature_pinscher',
    10: 'n02113712-miniature_poodle',
    11: 'n02102480-Sussex_spaniel',
    12: 'n02088364-beagle',
    13: 'n02105251-briard',
    14: 'n02099267-flat-coated_retriever',
    15: 'n02086240-Shih-Tzu',
    16: 'n02092339-Weimaraner',
    17: 'n02093428-American_Staffordshire_terrier',
    18: 'n02101388-Brittany_spaniel',
    19: 'n02105505-komondor'
 }

# fonction de prediction
def predict(img_path) :

    # chargement / preprocessing
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) 
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    # predictions
    predictions = MODEL.predict(img)

    # inverse one hot encoder => class 
    predictions = np.argmax(predictions, axis=1)

    return BREEDS.get(predictions[0]) 

if __name__ == "__main__":
    print(predict(sys.argv[1]))



# cmd : python.exe ./P06_VGG16_PREDICTION_SCRIPT.py "data\images\n02093256-Staffordshire_bullterrier\n02093256_269.jpg" 