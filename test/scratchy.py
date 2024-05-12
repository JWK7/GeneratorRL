import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2 
import pandas as pd
import scipy
import dmbrl.utils.DataFunctions as DataFunctions
import math


def ApplyGenerator(generator,I0):
    print(generator.shape)
    print(I0.shape)
    xGI0 = np.zeros(I0.shape)
    print(xGI0.shape)
    for i in range((I0.shape[1])):
        for j in range((I0.shape[2])):
            loc = np.matmul(generator,np.array(([[[j],[i],[1]]])))[:,:,0].astype(int)
            xGI0[:,loc[:,1]%I0.shape[1],loc[:,0]%I0.shape[2]] = I0[:,i,j]
    return xGI0

if __name__ == "__main__":
    by = 10/360*2*(math.pi)
    print(math.cos(by))
    generator = np.array([[math.cos(by),math.sin(by),0],[-math.sin(by),math.cos(by),0],[0,0,1]])
    generator = np.expand_dims(generator,0)
    img = cv2.imread("dmbrl/assets/cat.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(img_gray,(500,500))
    processed_img = np.expand_dims(processed_img, 0)

    Ix = ApplyGenerator(generator,processed_img)

    cv2.imwrite("outputs/originHorse.jpeg", Ix.reshape(500,500))