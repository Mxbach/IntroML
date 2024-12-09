# Implement the histogram equalization in this file
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


filepath = "./hello.png"

def main():
    img = Image.open(filepath).convert("L")
    data = np.array(img)
    
    row, col = data.shape
    # histogram
    hist = np.zeros((256))
    for i in range(row):
        for j in range(col):
            hist[data[i,j]] += 1

    # print(np.sum(hist[:90]))
    # normalized histogram
    norm_hist = hist / np.sum(hist)
 
    # cdf
    cdf = np.zeros(len(norm_hist))
    area = 0
    for i in range(len(norm_hist)):
        area += norm_hist[i]
        cdf[i] = area
    
    new_data = np.copy(data)

    c_min = np.min(cdf)
    for i in range(row):
        for j in range(col):
            new_data[i,j] = np.abs(((cdf[data[i,j]]-c_min)/(1-c_min))*255)
    
    kitty = Image.fromarray(new_data)
    kitty.save("kitty.png")
    
if __name__ == "__main__":
    main()