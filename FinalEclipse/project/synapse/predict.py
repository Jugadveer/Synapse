import numpy as np
import tensorflow as tf
from functools import lru_cache
from pathlib import Path
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128

classes = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]


@lru_cache(maxsize=1)
def get_model():
    model_path = Path(__file__).resolve().parent / "dementia_model.h5"
    return tf.keras.models.load_model(model_path)

def predict_mri(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = get_model().predict(img_array)
    class_idx = np.argmax(prediction)

    return classes[class_idx], float(np.max(prediction))