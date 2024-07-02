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
    print(Rx)
    print(Ry)
    for i in range(8):
        if i > len(Rx) - 1:
            continue
        dist += np.abs(Rx[i] - Ry[i])
    
    print(dist)
    return np.abs((1/8) * dist)


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    pass
