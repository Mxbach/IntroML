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
    arc = np.zeros((I, J), dtype=np.float64)
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
    
    # 
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
    pass


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
    # TODO
    pass


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
