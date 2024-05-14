from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    sample = np.load(filename)

    indices = np.argmax(np.abs(sample))
    # sample[indices]
    # sample = sample[:duration]
    # return sample[indices:]
    return sample[(indices+offset):((indices+offset))+duration]

def compute_frequency(signal, min_freq=20):
    sampling_rate = 44100
    n = len(signal)
    T_s = 1 / sampling_rate 
    f_n = np.fft.fft(signal)
    magnitude = np.abs(f_n)
    freqs = np.zeros(n)

    for i in range(n):
        if i <= n//2:
            freqs[i] = i / (n * T_s)
        else:
            freqs[i] = (i - n) / (n * T_s)
    
    mask = freqs > min_freq
    valid_magnitudes = magnitude[mask]
    valid_freqs = freqs[mask]
    peak_ind = np.argmax(valid_magnitudes)
    peak_f = valid_freqs[peak_ind]
    return peak_f

if __name__ == '__main__':
    # Implement the code to answer the questions here
    # s = load_sample("./sounds/Piano.ff.A4.npy")
    # c = compute_frequency(s)
    # print(c)
    # plt.plot(c)
    # plt.show()
    for i in range(2, 8):
        s = load_sample(f"./sounds/Piano.ff.A{i}.npy")
        c = compute_frequency(s)
        print(f"A{i}: " + str(c))
    s = load_sample("./sounds/Piano.ff.XX.npy")
    c = compute_frequency(s)
    print(f"XX: " + str(c)) #D6?

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
