import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2 
import pandas as pd
import scipy
import dmbrl.utils.DataFunctions as DataFunctions
def test2DImage():
    predG1 = np.reshape(pd.read_csv("outputs/G_1.csv",header=None).to_numpy(),(1,3*3*3,3*3*3))
    predG2 = np.reshape(pd.read_csv("outputs/G_2.csv",header=None).to_numpy(),(1,3*3*3,3*3*3))
    # predG2 = np.reshape(pd.read_csv("outputs/G_2.csv",header=None).to_numpy(),(400,400))
    
    # trueG = np.reshape(pd.read_csv("dmbrl/assets/Translation1D20Pixels.csv",header=None).to_numpy(),(7,7))

    predG1[predG1<-0.04] = -1
    predG1[predG1>0.04] = 1
    # predG1[predG1>-0.04 and predG1<0.04] = 0
    I0 = DataFunctions.NoiseImage((1000,3,3,3))
    # processed = np.zeros((3,3,3))
    # processed[5,5,:] = 240
    # futImage = np.matmul(scipy.linalg.expm((1 * predG2  )),processed)

    processed = I0.reshape(1000,3*3*3,1)
    futImage = np.matmul(-1* predG1  ,processed)+processed
    # futImage = np.matmul(0* predG2,np.matmul((30* predG1 ),processed))+ processed
    print(futImage.shape)

    rotated = np.zeros(I0.shape)
    for i in range(I0.shape[0]):
        rotated[i] = scipy.ndimage.rotate(I0[i], 1, axes=(1,0),reshape=True,mode = 'wrap')
        # rotated[i] = scipy.ndimage.rotate(I0[i], -30, axes=(2,1),reshape=False,mode = 'wrap')

    print(rotated[0,:,:,0])
    print(I0[0,:,:,0])
    exit()
    print(np.sum(np.abs(rotated-I0)))
    print(np.sum(np.abs(rotated.reshape(1000,3*3*3,1)-futImage)))

    # print(futImage)
    exit()
    # cv2.imwrite("outputs/originHorse1.jpeg", processed.reshape(3,3,3)[0,:,:])
    # cv2.imwrite("outputs/PredictedHorse1.jpeg", futImage.reshape(3,3,3)[0,:,:])
    # cv2.imwrite("outputs/originHorse2.jpeg", processed.reshape(3,3,3)[:,0,:])
    # cv2.imwrite("outputs/PredictedHorse2.jpeg", futImage.reshape(3,3,3)[:,0,:])
    # cv2.imwrite("outputs/originHorse3.jpeg", processed.reshape(3,3,3)[:,:,0])
    # cv2.imwrite("outputs/PredictedHorse3.jpeg", futImage.reshape(7,7,7)[:,:,0])

def Translation2DImage(I0,x,G):
    I0_ = np.reshape(I0,I0.shape[0:3])

    I0_[0] = ProcessImage((20,20))
    xs = np.random.randint(0,2,(I0.shape[0]))
    xs[xs==0] = -1
    xs = xs*x
    Ix = np.zeros(I0.shape)
    for _ in range(G.ndim): xs = np.expand_dims(xs,-1) 
    expxG = scipy.linalg.expm(xs*G)

    I0_ = np.matmul(expxG,I0_)
    Ix = np.matmul(I0_,expxG)
    Ix = np.expand_dims(Ix,(-1,-2))
    return xs,Ix




def ProcessImage(ImageSize: tuple):
    img = cv2.imread("dmbrl/assets/cat.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(img_gray,ImageSize)
    return processed_img


if __name__ == "__main__":
    test2DImage()