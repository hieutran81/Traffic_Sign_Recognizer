from LoadData import *
from sklearn.model_selection import train_test_split
import cv2
import datetime
import json
import pickle
from pprint import pprint
from PIL import Image
n_classes = 0
def splitData():
    X, y = readTrafficSigns()
    ar = np.asarray(X)
    dst = np.zeros([len(ar),32,32,3],dtype=float)
    for i in range(0,len(ar)):
        ar[i] = cv2.resize(ar[i],(32,32),0,0)
        dst[i,:,:,:] = ar[i]
    X_train, X_tc, y_train, y_tc = train_test_split(dst, y, test_size=0.3, random_state=42, shuffle=True)
    X_test, X_valid, y_test, y_valid = train_test_split(X_tc, y_tc, test_size=0.3, shuffle=True)
    global n_classes
    n_classes = np.unique(y_train).size
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def applyGrayscaleAndEqualizeHist(data):
    length = len(data)
    data = data.astype(np.float, copy=False)
    print("Applying Grayscale filter and Histogram Equalization")

    filteredData = []

    for data_sample in data[0:length, :]:
        data_sample = np.float32(data_sample)
        grayScale = cv2.cvtColor(data_sample, cv2.COLOR_BGR2GRAY)
        grayScale = np.uint8(grayScale)
        equalized = cv2.equalizeHist(grayScale)
        filteredData.append(np.reshape(equalized, (32, 32, 1)))

    return np.array(filteredData)

def normalize(data):
    length = len(data)
    data = data.astype(np.float, copy = False)

    print("Starting normalization: ", datetime.datetime.now().time())
    for data_sample in data[0:length, :]:
        for data_sample_row in data_sample:
            for data_sample_pixel in data_sample_row:
                data_sample_pixel[:] = [(color - 127.5) / 255.0 for color in data_sample_pixel]

    print("Normalization finished: ", datetime.datetime.now().time())
    return data

# preprocess before go to cnn
def preprocess():
    X_train, y_train, X_valid, y_valid, X_test, y_test = splitData()

    #load from json if exist
    # with open("data/train.p", "r") as f:
    #     X_train = pickle.load(f)
    # with open("data/test.p","r") as f:
    #     X_test = pickle.load(f)
    # with open("data/valid.p","r") as f:
    #     X_valid = pickle.load(f)
    X_train = applyGrayscaleAndEqualizeHist(X_train)
    X_valid = applyGrayscaleAndEqualizeHist(X_valid)
    X_test = applyGrayscaleAndEqualizeHist(X_test)
    # Normalization of pixel values from [0,255] to [-1,1]
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)

    # dump to json to fast
    # with open("data/train.p","w") as f:
    #     pickle.dump(X_train,f)
    # with open("data/test.p","w") as f:
    #     pickle.dump(X_test,f)
    # with open("data/valid.p","w") as f:
    #     pickle.dump(X_valid,f)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def getNumberClasses():
    global n_classes
    return n_classes

preprocess()



