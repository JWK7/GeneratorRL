import numpy as np
import scipy.ndimage
import cv2
import scipy

#Create Random Noise Image
def NoiseImage(image_size: tuple) -> np.ndarray:
    if type(image_size) is not tuple:
        print('NoiseImage Function Must Input Tuple')
        exit()
    return np.random.randint(0,255,image_size)

def Translation1DImage(I0: np.ndarray, x: float, G: np.ndarray) -> np.ndarray:
    xs = np.random.randint(0,2,(I0.shape[0],1,1))
    xs[xs==0] = -1
    xs = xs*x
    Ix = np.matmul(scipy.linalg.expm(xs*G),I0)
    return np.array([xs]),Ix

def Rotate2D(I0: np.ndarray,x: float):
    xs = np.random.randint(0,2,(I0.shape[0],1,1))
    xs[xs==0] = -1
    xs = xs*x

    rotated = np.zeros(I0.shape)
    for i in range(I0.shape[0]): rotated[i] = scipy.ndimage.rotate(I0[i], xs[i,0,0], axes=(1,0),reshape=False,mode = 'wrap')

    return np.array([xs]), rotated

def Rotate3D(I0: np.ndarray,x: float):
    xs1 = np.random.randint(0,2,(I0.shape[0],1,1))
    xs1[xs1==0] = -1
    xs1 = xs1*x

    xs2 = np.random.randint(0,2,(I0.shape[0],1,1))
    xs2[xs2==0] = -1
    xs2 = xs2*x

    xs3 = np.random.randint(0,2,(I0.shape[0],1,1))
    xs3[xs3==0] = -1
    xs3 = xs3*x

    rotated = np.zeros(I0.shape)
    for i in range(I0.shape[0]):
        rotated[i] = scipy.ndimage.rotate(I0[i], xs1[i,0,0], axes=(1,0),reshape=False,mode = 'wrap')
        rotated[i] = scipy.ndimage.rotate(I0[i], xs2[i,0,0], axes=(2,1),reshape=False,mode = 'wrap')
        rotated[i] = scipy.ndimage.rotate(I0[i], xs3[i,0,0], axes=(0,2),reshape=False,mode = 'wrap')
    return np.array([xs1,xs2,xs3]), rotated


def ProcessImage(ImageSize: tuple):
    img = cv2.imread("dmbrl/assets/cat.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(img_gray,ImageSize)
    return processed_img

def Translation2DImage(I0,x,G):
    I0_ = np.reshape(I0,I0.shape[0:3])

    I0_[0] = ProcessImage((20,20))
    xs1 = np.random.randint(0,2,(I0.shape[0]))
    xs1[xs1==0] = -1
    xs1 = xs1*x

    xs2 = np.random.randint(0,2,(I0.shape[0]))
    xs2[xs2==0] = -1
    xs2 = xs2*x

    Ix = np.zeros(I0.shape)
    for _ in range(G.ndim): 
        xs1 = np.expand_dims(xs1,-1) 
        xs2 = np.expand_dims(xs2,-1) 
    expxG1 = scipy.linalg.expm(xs1*G)
    expxG2 = scipy.linalg.expm(xs2*G)

    I0_ = np.matmul(expxG1,I0_)
    Ix = np.matmul(I0_,expxG2)
    Ix = np.expand_dims(Ix,(-1,-2))
    return np.array([xs1,xs2]),Ix


def Translation2DImageNoise(I0,x,G):
    I0_ = np.reshape(I0,I0.shape[0:3])

    I0_[0] = ProcessImage((20,20))
    xs1 = np.random.uniform(0,1,(I0.shape[0]))
    xs1[xs1==0] = -1
    xs1 = xs1*x

    xs2 = np.random.randint(0,1,(I0.shape[0]))
    xs2[xs2==0] = -1
    xs2 = xs2*x

    Ix = np.zeros(I0.shape)
    for _ in range(G.ndim): 
        xs1 = np.expand_dims(xs1,-1) 
        xs2 = np.expand_dims(xs2,-1) 
    expxG1 = scipy.linalg.expm(xs1*G)
    expxG2 = scipy.linalg.expm(xs2*G)

    I0_ = np.matmul(expxG1,I0_)
    Ix = np.matmul(I0_,expxG2)
    Ix = np.expand_dims(Ix,(-1,-2))
    return np.array([xs1,xs2]),Ix

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()