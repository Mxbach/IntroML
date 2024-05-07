from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    sample = np.load(filename)
    plt.plot(sample)
    plt.show()
    indices = np.argmax(sample)
    # sample[indices] += offset
    return sample[:duration]

def compute_frequency(signal, min_freq=20):
    # Complete this function
    return 0

if __name__ == '__main__':
    # Implement the code to answer the questions here
    s = load_sample("./sounds/Piano.ff.A2.npy")
    plt.plot(s)
    plt.show()

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
