from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import matplotlib.pyplot as plt
import scipy
import numpy as np

# TODO: Test the functions imported in lines 1 and 2 of this file.
# chirp = createChirpSignal(200, 1, 1, 10, True)
# triangle = createTriangleSignal(200, 2, 10000)
# square = createSquareSignal(200, 2, 10000)
saw = createSawtoothSignal(200, 2, 10000, 1)
plt.plot(saw)
plt.show()