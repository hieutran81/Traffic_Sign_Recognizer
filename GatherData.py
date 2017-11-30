import cv2
import matplotlib.pyplot as plt
import numpy as np


def rotateImage(image, angle):
    rows, cols = tuple(np.array(image.shape)[:2])
    image_center = tuple(np.array(image.shape)[:2]/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle, 1)
    result = cv2.warpAffine(image, rot_mat, (cols, rows),flags=cv2.INTER_LINEAR)
    return result

def translateImage(image, tx, ty):
    rows, cols = np.array(image.shape)[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def gatherData(image):
    images = []

    images.append(translateImage(image, 2, 0))
    images.append(translateImage(image, -2, 0))
    images.append(translateImage(image, 0, 2))
    images.append(translateImage(image, 0, -2))

    images.append(rotateImage(image, 10))
    images.append(rotateImage(image, -10))
    return images

if __name__ == "__main__":
    img = plt.imread("./data/train/00000/00000_00000.ppm")
    # print(type(img))
    print (img.shape)
    images = gatherData(img)
    # print(type(res))
    rows = 3
    cols = 3

    plt.subplot(rows,cols,1)
    plt.imshow(img)
    length = len(images)
    print(length)
    for i in range(2, cols*rows + 1):
        if (i - 2 == length):
            break
        plt.subplot(rows,cols, i)
        plt.imshow(images[i-2])
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()
    labels = [1, 1]
    labels.extend([5] * 0)
    print(labels)


