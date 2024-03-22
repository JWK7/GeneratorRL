import numpy as np
import cv2 
import pandas as pd
import scipy
def test2DImage():
    predG = np.reshape(pd.read_csv("outputs/G_1.csv",header=None).to_numpy(),(400,400))
    processed = ProcessImage((20,20)).flatten()
    futImage = np.matmul(scipy.linalg.expm(-5*(predG)),processed)

    cv2.imwrite("outputs/originHorse.jpeg", processed.reshape(20,20))
    cv2.imwrite("outputs/PredictedHorse.jpeg", futImage.reshape(20,20))
    




def ProcessImage(ImageSize: tuple):
    img = cv2.imread("dmbrl/assets/cat.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(img_gray,ImageSize)
    return processed_img


if __name__ == "__main__":
    test2DImage()