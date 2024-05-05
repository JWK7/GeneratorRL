import numpy as np
import cv2 
import pandas as pd
import scipy
def test2DImage():
    predG1 = np.reshape(pd.read_csv("outputs/G_1.csv",header=None).to_numpy(),(400,400))

    predG2 = np.reshape(pd.read_csv("outputs/G_2.csv",header=None).to_numpy(),(400,400))
    
    trueG = np.reshape(pd.read_csv("dmbrl/assets/Translation1D20Pixels.csv",header=None).to_numpy(),(20,20))


    processed = ProcessImage((20,20)).flatten()
    # futImage = np.matmul(scipy.linalg.expm((0* predG1+ 1 * predG2  )),processed)

    futImage = np.matmul((0* predG1+ 2 * predG2 ),processed)+ processed

    expxG = scipy.linalg.expm(-1*trueG)
    print(expxG.shape)
    print(processed.shape)
    I0_ = np.matmul(expxG,processed.reshape(20,20))
    Ix = np.matmul(I0_,expxG)
    Ix = np.expand_dims(Ix,(-1,-2))

    cv2.imwrite("outputs/originHorse.jpeg", processed.reshape(20,20))
    cv2.imwrite("outputs/PredictedHorse.jpeg", futImage.reshape(20,20))
    cv2.imwrite("outputs/trueHorse.jpeg", Ix.reshape(20,20))
    
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