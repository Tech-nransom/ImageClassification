import cv2
import tensorflow as tf

Catagories = ["darth_vader", 'cat']



def prepare(filepath):
    IMG_SIZE = 60
    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    return new_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("final.h5")


classes = model.predict([prepare('C:/Users/yugandhar yelai/PycharmProjects/tensorEnv/ImageClassification/test.jpg')])

print(classes)
