import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")



class TFHubWrapper(tf.keras.layers.Layer):
    def __init__(self, hub_url=None, **kwargs):
        super().__init__(**kwargs)
        if hub_url:
            self.hub_layer = hub.KerasLayer(hub_url, trainable=False)
        else:
            self.hub_layer = None

    def call(self, inputs):
        return self.hub_layer(inputs)



def load_model_file():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TFHubWrapper": TFHubWrapper, "KerasLayer": hub.KerasLayer}
    )


def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ---------------------- PREDICT ----------------------
def predict_breed(model, class_names, image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx], preds
