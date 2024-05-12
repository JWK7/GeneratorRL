import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2 
import pandas as pd
import scipy
import dmbrl.utils.DataFunctions as DataFunctions



if __name__ == "__main__":
    I0 = DataFunctions.NoiseImage(((100,)+(10,10,10)))

    print(I0.shape)
    cv2.imwrite("outputs/rotprex.jpeg",I0[10,5,:,:])
    cv2.imwrite("outputs/rotprey.jpeg",I0[10,:,5,:])
    cv2.imwrite("outputs/rotprez.jpeg",I0[10,:,:,5])
    x,Ix = DataFunctions.Rotate3D(I0,90)
    cv2.imwrite("outputs/rotx.jpeg", Ix[10,5,:,:])
    cv2.imwrite("outputs/roty.jpeg", Ix[10,:,5,:])
    cv2.imwrite("outputs/rotz.jpeg", Ix[10,:,:,5])