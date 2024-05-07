import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    l = []
    step = 1 / samples
    for t in [i*step for i in range(samples)]:
        s = 0
        for k in range(k_max+1):
            s += (-1)**k *((np.sin(2*np.pi*(2*k+1)*frequency*t))/(2*k+1)**2)
        l.append((8/(np.pi**2)) * s)
    return np.array(l)


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    l = []
    step = 1 / samples
    for t in [i*step for i in range(samples)]:
        s = 0
        for k in range(1, k_max+1):
            s += (np.sin(2*np.pi*(2*k-1)*frequency*t))/(2*k-1)
        l.append(4/(np.pi) * s)
    return np.array(l)


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    l = []
    step = 1 /samples
    for t in [i*step for i in range(samples)]:
        s = 0
        for k in range(1, k_max+1):
            s += np.sin(2*np.pi*k*frequency*t)/k
        l.append((amplitude/2) - ((amplitude/np.pi)*s))

    return np.array(l)
