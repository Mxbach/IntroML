import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    step = duration/samplingrate
    if linear:
        c = (freqto - freqfrom)/duration
        return np.array([np.sin(2*np.pi * (freqfrom + (c/2)*t)*t) for t in [i*step for i in range(samplingrate)]])
    k = (freqto/freqfrom)**(1/duration)
    return np.array([np.sin(((2*np.pi*freqfrom)/np.log(k))*(k**t - 1)) for t in [i*step for i in range(samplingrate)]])