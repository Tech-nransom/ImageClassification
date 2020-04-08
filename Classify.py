import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random
import pickle

DATADIR = "F:\DATA"  # File Directory Path
CATAGORIES = ["darth_vader", "cat"]  # Give Exact Same Name As Folder Names

for catagory in CATAGORIES:
    path = os.path.join(DATADIR, catagory)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap="gray")
        plt.show()
        break
    break

print(img_arr)

IMG_SIZE = 60
new_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap="gray")
plt.show()

training_data = []


def createTrainingData():
    for catagory in CATAGORIES:
        path = os.path.join(DATADIR, catagory)
        class_num = CATAGORIES.index(catagory)
        for img in os.listdir(path):
            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


createTrainingData()
print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []  # Features
y = []  # Lables

for features, lables in training_data:
    X.append(features)
    y.append(lables)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  #1 becz of Grey scale


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


