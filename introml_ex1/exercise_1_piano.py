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
    # freqs = np.fft.fftfreq(n, T_s)
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
    d = {"A2": 110.0000, "A3": 220.0000, "A4": 440.0000, "A5": 880.0000, "A6": 1760.000, "A7": 3520.000, "D6": 1174.659}
    diffs = []
    print("--------------------------")
    for i in range(2, 8):
        s = load_sample(f"./sounds/Piano.ff.A{i}.npy")
        c = compute_frequency(s)
        original_val = d[f"A{i}"]
        difference = np.abs(c-original_val)
        diffs.append(difference)
        print(f"A{i}: {str(c)}\nOriginal: {original_val}\nDiffernce: {difference}")
        print("--------------------------")

        if i == 5:
            s = load_sample("./sounds/Piano.ff.XX.npy")
            c = compute_frequency(s)
            original_val = d["D6"]
            difference = np.abs(c-original_val)
            diffs.append(difference)
            print(f"XX: {str(c)}\nOriginal (D6): {original_val}\nDifference: {difference}")
            print("--------------------------")
    
    plt.plot(np.array(diffs))
    plt.show()
    """
    s = load_sample("./sounds/Piano.ff.XX.npy")
    c = compute_frequency(s)
    original_val = d["D6"]
    print("XX")
    print(f"D6: {str(c)}\nOriginal: {original_val}\nDifference: {np.abs(c-original_val)}") #D6?
    """
    

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
