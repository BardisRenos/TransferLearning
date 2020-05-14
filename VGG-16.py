import random
from keras.applications.vgg16 import VGG16
import numpy as np
import os
import cv2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import np_utils

# The size that we want to reformat all the images in the same dimensions
IMG_SIZE = 250


def create_training_data() -> list:
    # Here we set the path of the dataset.
    DATALOCATION = 'C:\\path\\of\\the\\file'
    # The categories of the data set. Are two only Cats and dogs. Hence, Dog is category 0 and the Cat is 1
    CATEGORIES = ["Dog", "Cat"]
    training_data = []
    for categories in CATEGORIES:
        path = os.path.join(DATALOCATION, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

    return training_data


def creating_features_labels():
    # Data and labels
    X = []
    y = []
    data = create_training_data()
    # We shuffle the data to be random
    random.shuffle(data)
    for features, label in data:
        X.append(features)
        y.append(label)
    # We reshape the data in order to fit the DNN model
    X = np.asarray(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X = X.astype('float32')
    y = np.array(y, dtype=np.float32)
    X /= 255.0
    y = np_utils.to_categorical(y)
    return X, y


def train_model():
    X, y = creating_features_labels()

    # Load the VGG model without the last layers
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    print(vgg_model.summary())
    # We can add the last part the classier layers
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    # Full connected layer with 1024 neurons
    x = Dense(1024, activation='relu')(x)  # dense layer 1
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(4096, activation='relu')(x)  # dense layer 3
    # We change the output leyar to 2 classes from 1000 classes before
    output = Dense(2, activation='softmax')(x)

    # We can Define the new model
    model = Model(inputs=vgg_model.input, outputs=output)

    # We stop the layers to be trained
    for layer in model.layers[:-3]:
        layer.trainable = False

    print(model.summary())

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, validation_split=0.1, epochs=5, batch_size=32)

    # Final evaluation of the model
    scores = model.evaluate(X, y)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    train_model()
