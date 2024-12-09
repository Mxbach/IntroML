import numpy as np
from matplotlib import pyplot as plt
#
# NO OTHER IMPORTS ALLOWED
#

def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    return np.histogram(img, 256, (0, 255))[0]


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    first = np.where((img <= t), img, 255)
    second = np.where((first > t), first, 0)
    return second


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    sum = np.sum(hist)
    return np.sum(hist[:theta+1])/sum, np.sum(hist[theta+1:])/sum


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: normalized histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    m0 = 0
    m1 = 0
    for i in range(theta+1):
        m0 += i * hist[i]
    for i in range(theta+1, len(hist)):
        m1 += i * hist[i]
    
    m0 = m0/p0 if p0 > 0 else 0
    m1 = m1/p1 if p1 > 0 else 0

    return m0, m1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    p0 = 0
    p1 = 0
    m0 = 0
    m1 = 0
    sig_int = 0
    thresh = 0
    threshs = []
    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    norm_hist = hist / np.sum(hist)
    # TODO loop through all possible thetas
    for theta in range(len(hist)):
        # TODO compute p0 and p1 using the helper function
        p0, p1 = p_helper(hist, theta)
        # TODO compute mu and m1 using the helper function
        m0, m1 = mu_helper(norm_hist, theta, p0, p1)
        # TODO compute variance
        sig_int = p0 * p1 * ((m1 - m0)**2)

        # TODO update the threshold
        threshs.append(sig_int)
    
    return np.argmax(threshs)


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    hist = create_greyscale_histogram(img)
    t = calculate_otsu_threshold(hist)
    return binarize_threshold(img, t)