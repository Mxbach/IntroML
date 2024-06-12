import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from convo import make_kernel


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    k = make_kernel(ksize, sigma)
    con = convolve(img_in, k)
    return (k, con.astype(int))


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
    # ist vertikal geflippt
    gy = [[1 , 2, 1], [0, 0, 0], [-1, -2, -1]]

    return convolve(img_in, gx).astype(int), convolve(img_in, gy).astype(int)


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO

    I, J = gx.shape
    g = np.zeros((I, J), dtype=int)
    # auf np.float ändern
    arc = np.zeros((I, J), dtype=np.float)
    for i in range(I):
        for j in range(J):
            g[i, j] = int(np.sqrt(gx[i, j]**2 + gy[i, j]**2))
            arc[i,j] = np.arctan2(gy[i,j], gx[i,j])

    # return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx).astype(int)
    return g, arc


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    degree = angle * 180/np.pi
    degree = degree % 180
    # Possible matching angles in degrees
    l = [0, 45, 90, 135, 180]
    t = 0
    v = [(np.abs(x - degree)) for x in l]
    t = l[len(v) - 1 - np.argmin(v[::-1])]
    
    return 0 if t == 180 else t

    
def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    I, J = g.shape
    out = np.zeros((I, J))
    dir = {90: [(1, 0), (-1, 0)], 45: [(1, -1), (-1, 1)], 0: [(0, 1), (0, -1)], 135: [(1, 1), (-1, -1)]}
    for i in range(I):
        for j in range(J):
            t = convertAngle(theta[i, j])

            x1, y1 = dir[t][0] 
            x1, y1 = (0, 0) if (not (0 <= i + x1 < I) or not (0 <= j + y1 < J)) else (x1, y1)

            x2, y2 = dir[t][1]
            x2, y2 = (0, 0) if (not (0 <= i + x2 < I) or not (0 <= j + y2 < J)) else (x2, y2)

            if  (g[i, j] >= g[i + x1, j + y1] and g[i, j] >= g[i + x2, j + y2]):
                out[i, j] = g[i, j]

    return out

def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    I, J = max_sup.shape
    thres_img = np.zeros((I, J))
    out = np.copy(max_sup)

    for i in range(I):
        for j in range(J):
            if max_sup[i, j] <= t_low:
                thres_img[i, j] = 0
            elif t_low < max_sup[i, j] and max_sup[i, j] <= t_high:
                thres_img[i, j] = 1
            elif max_sup[i, j] > t_high:
                thres_img[i, j] = 2
    
    neighbours = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for i in range(I):
        for j in range(J):
            if thres_img[i, j] == 2:
                out[i, j] = 255
            
                for x, y in neighbours:
                    if (0 <= i + x <= I-1) and (0 <= j + y <= J-1):
                        if thres_img[i + x, j + y] == 1:
                            out[i + x, j + y] = 255
                        
    return out

def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
