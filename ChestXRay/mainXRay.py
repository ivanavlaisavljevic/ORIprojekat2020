import numpy as np
from pathlib import Path
import csv

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPool2D, AveragePooling2D, MaxPooling3D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import Activation
from PIL import Image
import cv2
import keras

img_rows, img_cols = 64, 64

epochs = 70
batch_size = 32
num_classes = 3
correct_answer = {}
answer = ["BACTERIA", "NORMAL", "VIRUS"]

train_dir = Path("data/train")
val_dir = Path("data/val")

if __name__ == '__main__':
    print("pocelo")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'data/val',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical')

    # MODEL

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.01)

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt, metrics=['accuracy'])

    hist = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=epochs)
    # Show results
    score = model.evaluate_generator(train_generator, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights("trying.h5")

    # print("[ LOADING WEIGHTS... ]")
    # model.load_weights("trying.h5")
    # print("[ WEIGHTS LOADED! ]")
    #
    # with open('metadata.csv', mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #         if row["Label"] == "Pnemonia":
    #             correct_answer[row["X_ray_image_name"]] = row["Label_1_Virus_category"].upper()
    #         else:
    #             correct_answer[row["X_ray_image_name"]] = row["Label"].upper()
    #
    #
    # img_names = correct_answer.keys()
    #
    # img_to_predict = None
    # correct = 0
    # incorrect = 0
    #
    # for name in img_names:
    #     path_img_to_predict = "data\\test\\" + name
    #     img_to_predict = cv2.imread(path_img_to_predict)
    #     img_to_predict = cv2.resize(img_to_predict, (img_rows, img_cols))
    #     img_to_predict = np.reshape(img_to_predict, (1, img_rows, img_cols, 3))
    #     res = answer[model.predict_classes(np.array(img_to_predict))[0]]
    #
    #     if res == correct_answer[name]:
    #         correct = correct + 1
    #     else:
    #         incorrect = incorrect + 1
    #
    # print("[CORRECT ANSWERS: ", correct, "]\n[INCORRECT ANSWERS: ", incorrect, "]")
    # perc = correct / (correct + incorrect) * 100
    # print("[PERCENTAGE OF ACCURACY: ", perc, "]")


