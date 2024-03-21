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