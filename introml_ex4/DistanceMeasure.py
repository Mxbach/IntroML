'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    dist = 0
    min_len = min(len(Rx), len(Ry))
    for i in range(8):
        if i > len(Rx) - 1:
            continue
        dist += np.abs(Rx[i] - Ry[i])
    
    return dist / min_len


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''

    lx_sum = np.sum(Thetax[:8])
    ly_sum = np.sum(Thetay[:8])

    lxx = 0
    lyy = 0
    lxy = 0
    for i in range(8):
        if i < len(Thetax):
            lxx += (Thetax[i] - (1/len(Thetax)) * lx_sum)**2
        if i < len(Thetay):
            lyy += (Thetay[i] - (1/len(Thetay)) * ly_sum)**2
        if i < len(Thetax) and i < len(Thetay):
            lxy += (Thetax[i] - (1/len(Thetax)) * lx_sum) * (Thetay[i] - (1/len(Thetay)) * ly_sum)

    dist = ( 1 - ( (lxy*lxy) / (lxx*lyy) ) ) * 100

    return dist