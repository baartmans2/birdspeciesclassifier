import tensorflow as tf
import data
import os
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from typing import List

INITIAL_EPOCHS = 10

CHECKPOINT_DIR = "training/"
CHECKPOINT_PATH = CHECKPOINT_DIR + "cp.ckpt"
CHECKPOINT_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

TEST_PATH = 'images/domestic_chicken.jpeg'


def create_model():
    base_model = tf.keras.applications.InceptionV3(include_top=False,)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling_layer")(x)
    outputs = tf.keras.layers.Dense(
        525, activation="softmax", name="output-layer")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["accuracy"],)
    return model


def load_model(model: tf.keras.Model):
    model.load_weights(CHECKPOINT_PATH).expect_partial()
    return model


def train_model(model: tf.keras.Model):
    history = model.fit(data.TRAIN_DATA, epochs=INITIAL_EPOCHS, steps_per_epoch=len(data.TRAIN_DATA),
                        validation_data=data.VALID_DATA, validation_steps=int(0.25*len(data.VALID_DATA)), callbacks=[CHECKPOINT_CALLBACK])
    return history


def test_model(model: tf.keras.Model):
    model.evaluate(data.TEST_DATA)


def tune_model(model: tf.keras.Model, history, tune_epochs):
    model.trainable = True

    for layer in model.layers[:-10]:
        layer.trainable = False

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), metrics=["accuracy"])

    new_epochs = INITIAL_EPOCHS + tune_epochs
    model.fit(data.TRAIN_DATA, epochs=new_epochs, steps_per_epoch=len(data.TRAIN_DATA),
              validation_data=data.VALID_DATA, validation_steps=int(0.25*len(data.VALID_DATA)), callbacks=[CHECKPOINT_CALLBACK], initial_epoch=history.epoch[-1])


def classify(model: tf.keras.Model, img_path):
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(
        img_batch)
    prediction = model.predict(img_preprocessed).tolist()[0]
    print(data.get_bird_name(prediction.index(1.0)))


model = create_model()
if os.path.exists(CHECKPOINT_DIR):
    model = load_model(model)
else:
    history = train_model(model)
model.summary()
data.plot_curves(model.history)
classify(model, TEST_PATH)
