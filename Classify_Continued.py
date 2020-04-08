import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import pickle
import time
import cv2

CATAGORIE = ["darth_vader", "cat"]

'''
# print(model)
# prediction = model.predict([prepare("trial.png")])
# print(prediction)


NAME = "DarthVsCats_{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs{}'.format(NAME))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              matrics=["accuracy"])

IMG_SIZE = 60

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = np.array(y)

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

model.save("final.h5")

'''


def prepare(filepath):
    IMG_SIZE = 60
    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    return new_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("final.h5")

'''
model.compile(loss='binary_crossentropy',
              optimizer='adam',

              metrics=['accuracy'])
'''
'''
IMG_SIZE = 60

img = cv2.imread('trial.png')
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
fi = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
'''
image = tf.image.decode_jpeg("test.jpg")
image = tf.cast(image, tf.float32)

classes = model.predict([prepare(image)])

print(classes)
