'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = (ksize//2, ksize//2)
    for i in range(ksize):
        for j in range(ksize):
            x = np.abs(i - center[0])
            y = np.abs(j - center[1])
            kernel[i, j] = 1/(2*np.pi* sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    
    return kernel / np.sum(kernel) # implement the Gaussian kernel here

def slow_convolve(arr, k):

    I, J = arr.shape
    U, V = k.shape
    
    # first = np.flip(k, axis=0)
    # kernel = np.flip(first, axis=1)
    kernel = k[::-1, ::-1] 

    lr = int(np.floor(U/2))
    tb = int(np.floor(V/2))

    new_arr = np.zeros((I, J))
    
    enlarged = np.zeros((I + 2*lr, J + 2*tb))
    enlarged[lr : lr+I, tb : tb+J] = arr[:, :]

    for i in range(I):
        for j in range(J):
            sum = 0
            for u in range(- lr, int(np.ceil(U/2))):
                for v in range(- tb, int(np.ceil(V/2))):
                    sum += kernel[u + lr, v + tb] * enlarged[i + u + lr, j + v + tb]
            new_arr[i, j] = sum
    return new_arr

def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    bin = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)[1]
    kernel = make_kernel(5, 1)
    res = cv2.filter2D(bin, -1, kernel)
    # print(res)
    return res


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(img)
    # mask = out == 0
    # out = np.where(mask, out, 255)
    cv2.drawContours(out, [c], -1, 255, 2)

    return out


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    I, J = contour_img.shape
    intersections = []
    con = 255
    back = 0

    for i in range(I):
        if i == 0 or i == I-1:
            continue
        elif contour_img[i, x] == con:
            if contour_img[i-1, x] == back or contour_img[i+1, x] == back:
                intersections.append(i+1)

    # print("Contour image:")
    # with np.printoptions(threshold=np.inf):
    #     print(contour_img)
    # 
    # print("Intersections:", intersections)
    out = intersections[1::2]
    # out.extend(intersections[-5:-2])
    # print(out[:6])
    return np.array(out[:6])


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    m = (y2 - y1) / (x2 - x1)
    t = y2 - m * x2

    Y, X = img.shape
    for x in range(X):
        y = int(m*x + t)
        if y < Y:
            if img[y, x] == 255:
                return (y, x)
    # [y, x]


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    y1, x1 = k1
    y2, x2 = k2
    y3, x3 = k3

    my = (y3 - y1) / (x3 - x1)
    ty = y3 - my * x3

    mx = -1/(my)
    tx = y2 - mx * x2
    
    div =  (1 + mx * my) if (1 + mx * my) != 0 else 1

    angle = np.arctan(mx)

    nx = int(np.round((tx - ty) / (my - mx)))
    ny = int(np.round(my * nx  + ty))
    return cv2.getRotationMatrix2D((ny, nx), np.degrees(angle), 1)



def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
	
    # TODO threshold and blur
    blur = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    con = drawLargestContour(blur)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 5
    x2 = 15
    intersections1 = getFingerContourIntersections(con, x1)
    intersections2 = getFingerContourIntersections(con, x2)
    
    # TODO compute middle points from these contour intersections
    y11 = (intersections1[0] + intersections1[1]) / 2
    y12 = (intersections1[2] + intersections1[3]) / 2
    y13 = (intersections1[4] + intersections1[5]) / 2

    y21 = (intersections2[0] + intersections2[1]) / 2
    y22 = (intersections2[2] + intersections2[3]) / 2
    y23 = (intersections2[4] + intersections2[5]) / 2

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(con, y11, x1, y21, x2)
    k2 = findKPoints(con, y12, x1, y22, x2)
    k3 = findKPoints(con, y13, x1, y23, x2)

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    rot = getCoordinateTransform(k1, k2, k3)
    
    # TODO rotate the image around new origin
    out = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]))
    
    return out
