'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!

# 1. np.fft
# 2. np.fftshift
# 3. Calculate dB from Fourier Transform F: 20 * log_10(F)

def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    height, width = shape
    
    center_y = height / 2
    center_x = width / 2
    
    y = center_y + r * np.sin(theta)
    x = center_x + r * np.cos(theta)

    return y, x



def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    Return Magnitude in Decibel
    :param img:
    :return:
    '''
    mag = np.abs(np.fft.fft2(img))
    shift = np.fft.fftshift(mag)

    out = 20 * np.log10(shift)
    return out


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring --> theta/sampling rate
    :return: feature vector of k features
    '''
    # print(magnitude_spectrum)
    l = []
    height, width = magnitude_spectrum.shape
    # with np.printoptions(threshold=np.inf):
    #     print(magnitude_spectrum)
    #
    # print("shape", magnitude_spectrum.shape)
    for i in range(1,k+1):
        sum = 0
        for step in range(sampling_steps):
            theta = step * (np.pi / (sampling_steps-1)) # -1 ?
            # print(theta)
            for r in range(8 * (i - 1) + 1, (8 * i)): # vllt + 1 - 1
                # print("polar", theta, r)
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                # print("kart", y, x)
                if 0 <= int(y) < height and 0 <= int(x) < width:
                    # print(magnitude_spectrum[int(np.round(y)), int(np.round(x))])
                    sum += magnitude_spectrum[int(y), int(x)]
        l.append(sum)
    print(l)
    return np.array(l)


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area --> theta/sampling rate
    :return: feature vector of length k
    """
    I, J = magnitude_spectrum.shape
    length = I if I < J else J

    l = []
    for i in range(k):
        sum = 0
        for steps in range(sampling_steps):
            theta = steps * (length / sampling_steps)
            for r in range(length):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta*np.pi / k-1)
                #print(y, x)
                sum += magnitude_spectrum[int(y), int(x)]
        
        l.append(sum)

    return np.array(l)

def calcuateFourierParameters(img, k, sampling_steps) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    pass
