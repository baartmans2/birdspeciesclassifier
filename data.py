from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pyplot
import os
from typing import List

DIR = "Data/"
TRAIN_DIR = DIR + "train/"
TEST_DIR = DIR + "test/"
VALID_DIR = DIR + "valid/"

TRAIN_GEN = ImageDataGenerator(rescale=1./255)
TEST_GEN = ImageDataGenerator(rescale=1./255)
VALID_GEN = ImageDataGenerator(rescale=1./255)

TRAIN_DATA = TRAIN_GEN.flow_from_directory(
    TRAIN_DIR, batch_size=32, target_size=(224, 224), class_mode="categorical")
TEST_DATA = TEST_GEN.flow_from_directory(
    TEST_DIR, batch_size=32, target_size=(224, 224), class_mode="categorical")
VALID_DATA = VALID_GEN.flow_from_directory(
    TRAIN_DIR, batch_size=32, target_size=(224, 224), class_mode="categorical")

BIRD_CLASSES = sorted(os.listdir(TRAIN_DIR))


def get_bird_name(index):
    return BIRD_CLASSES[index]


def plot_curves(history):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # plot loss
    pyplot.plot(epochs, loss, label="training_loss")
    pyplot.plot(epochs, val_loss, label="val_loss")
    pyplot.title("loss")
    pyplot.xlabel("epochs")
    pyplot.legend()

    # plot accuracy
    pyplot.figure()
    pyplot.plot(epochs, accuracy, label="training_accuracy")
    pyplot.plot(epochs, val_accuracy, label="val_accuracy")
    pyplot.title("accuracy")
    pyplot.xlabel("epochs")
    pyplot.legend()
