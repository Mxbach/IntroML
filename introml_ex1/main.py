from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import matplotlib.pyplot as plt
#import scipy
import numpy as np
# TODO: Test the functions imported in lines 1 and 2 of this file.
#------------exercise1 -----------------
#
# chirp = createChirpSignal(200, 1, 1, 10, True)
#
#------------exercise2------------------
#
samples = 200
frequency = 2
kMax = 10000
amplitude = 1
#
# triangle = createTriangleSignal(samples, frequency, kMax)
# square = createSquareSignal(samples, frequency, kMax)
saw = createSawtoothSignal(samples, frequency, kMax, amplitude)
#
plt.plot(saw)
plt.xticks(ticks=plt.xticks()[0][1:-1], labels=(1/samples) * np.array(plt.xticks()[0][1:-1], dtype=np.float64))
plt.show()
