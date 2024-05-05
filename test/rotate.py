import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2 
import pandas as pd
import scipy
import dmbrl.utils.DataFunctions as DataFunctions



if __name__ == "__main__":
    I0 = DataFunctions.NoiseImage(((100,)+(20,20)))

    cv2.imwrite("outputs/rotpre.jpeg",I0[10])
    x,Ix = DataFunctions.rotate(I0,90)
    cv2.imwrite("outputs/rot.jpeg", Ix[10])