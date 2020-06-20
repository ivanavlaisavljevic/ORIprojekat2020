import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
import keras

img_rows, img_cols = 224, 224

epochs = 12
batch_size = 16
num_classes = 3

train_dir = Path("data/train")
val_dir = Path("data/val")

if __name__ == '__main__':
    print("pocelo")

    normal_cases_dir = train_dir / 'NORMAL'
    bacteria_cases_dir = train_dir / 'BACTERIA'
    virus_cases_dir = train_dir / 'VIRUS'

    # Get the list of all the images
    normal_cases = normal_cases_dir.glob('*.jpeg')
    bacteria_cases = bacteria_cases_dir.glob('*.jpeg')
    virus_cases = virus_cases_dir.glob('*.jpeg')


    # List that are going to contain validation images data and the corresponding labels
    train_data = []
    train_labels = []

    # Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.
    # We will normalize the pixel values and resizing all the images to 224x224



    # Normal cases
    i = 0
    for img in normal_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(0, num_classes=3)
        train_data.append(img)
        train_labels.append(label)
        i = i + 1
    print("############################## ", i)

    # Bacteria cases
    for img in bacteria_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(1, num_classes=3)
        train_data.append(img)
        train_labels.append(label)

    # Virus cases
    for img in virus_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(2, num_classes=3)
        train_data.append(img)
        train_labels.append(label)

    # Convert the list into numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    if K.image_data_format() == 'channels_first':
        train_data = train_data.reshape(len(train_data), 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        train_data = train_data.reshape(len(train_data), img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)


    ####################VALIDATION#####################
    normal_cases_dir = val_dir / 'NORMAL'
    bacteria_cases_dir = val_dir / 'BACTERIA'
    virus_cases_dir = val_dir / 'VIRUS'

    # Get the list of all the images
    normal_cases = normal_cases_dir.glob('*.jpeg')
    bacteria_cases = bacteria_cases_dir.glob('*.jpeg')
    virus_cases = virus_cases_dir.glob('*.jpeg')

    # List that are going to contain validation images data and the corresponding labels
    test_data = []
    test_labels = []

    # Normal cases
    for img in normal_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(0, num_classes=3)
        test_data.append(img)
        test_labels.append(label)

    # Bacteria cases
    for img in bacteria_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(1, num_classes=3)
        test_data.append(img)
        test_labels.append(label)

    # Virus cases
    for img in virus_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.
        label = to_categorical(2, num_classes=3)
        test_data.append(img)
        test_labels.append(label)

    # Convert the list into numpy arrays
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(len(test_data), 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        test_data = test_data.reshape(len(test_data), img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    # model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # kompajliranje modela za multiklasnu klasifikaciju
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # treniranje i evaluacija
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(test_data, test_labels))
    # ispis rezultata
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

