import numpy as np
import scipy.ndimage
import cv2
import scipy

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


def ProcessImage(ImageSize: tuple):
    img = cv2.imread("dmbrl/assets/cat.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(img_gray,ImageSize)
    return processed_img

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


@staticmethod
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