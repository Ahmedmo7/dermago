from PIL import Image
import numpy as np
import io
import base64
import numpy as np
import io
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras import backend as K
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import flask
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


def get_model():
    global model
    model = tf.keras.models.load_model('model2InceptionV3.hdf5')
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras Model...")
get_model()


@app.route("/", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(150, 112))

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'bkl': prediction[0][0],
            'nv': prediction[0][1],
            'df': prediction[0][2],
            'mel': prediction[0][3],
            'vasc': prediction[0][4],
            'bcc': prediction[0][5],
            'akiec': prediction[0][6]
        }
    }

    return jsonify(response)

