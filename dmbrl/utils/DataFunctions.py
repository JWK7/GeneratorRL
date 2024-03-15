import numpy as np
import scipy.ndimage

#Create Random Noise Image
@staticmethod
def NoiseImage(image_size: tuple) -> np.ndarray:
    if type(image_size) is not tuple:
        print('NoiseImage Function Must Input Tuple')
        exit()
    return np.random.randint(0,255,image_size)

@staticmethod
def Translation1DImage(I0: np.ndarray, x: float, G: np.ndarray) -> np.ndarray:
    xs = np.random.randint(0,2,(I0.shape[0],1,1))
    xs[xs==0] = -1
    xs = xs*x
    Ix = np.matmul(scipy.linalg.expm(xs*G),I0)
    return xs,Ix
