from PIL import Image
import numpy as np
from scipy.signal import convolve


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = (ksize//2, ksize//2)
    for i in range(ksize):
        for j in range(ksize):
            x = np.abs(i - center[0])
            y = np.abs(j - center[1])
            kernel[i, j] = 1/(2*np.pi* sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    
    return kernel / np.sum(kernel) # implement the Gaussian kernel here

def slow_convolve(arr, k):

    I, J = arr.shape
    U, V = k.shape
    
    # first = np.flip(k, axis=0)
    # kernel = np.flip(first, axis=1)
    kernel = k[::-1, ::-1] 

    lr = int(np.floor(U/2))
    tb = int(np.floor(V/2))

    new_arr = np.zeros((I, J))
    
    enlarged = np.zeros((I + 2*lr, J + 2*tb))
    enlarged[lr : lr+I, tb : tb+J] = arr[:, :]

    for i in range(I):
        for j in range(J):
            sum = 0
            for u in range(- lr, int(np.ceil(U/2))):
                for v in range(- tb, int(np.ceil(V/2))):
                    sum += kernel[u + lr, v + tb] * enlarged[i + u + lr, j + v + tb]
            new_arr[i, j] = sum
    return new_arr

"""
def slow_convolve(arr, k):
    I, J = arr.shape
    new_arr = np.zeros((I, J))

    U, V = k.shape
    kernel = k[::, ::]

    lr = int(np.floor(U/2))
    tb = int(np.floor(V/2))

    enlarged = np.zeros((I + 2*lr, J + 2*tb))
    enlarged[lr : lr+I, tb : tb+J] = arr[:, :]

    print(f"kernel:\n{kernel}\n")   
    print(f"enlarged:\n{enlarged}\n")   
    print(f"U: {U}, V: {V}")
    for i in range(I):
        for j in range(J):
            sum = 0
            # -1 bis 2 ex ## u = 1
            for u in range(- int(np.floor(U/2)), int(np.ceil(U/2))): # U = 3, U/2 = 1,5
                # -1 bis 1 ex ## v = -1
                for v in range(- int(np.floor(V/2)), int(np.ceil(V/2))): # V = 2, V/2 = 1
                    sum += kernel[u + int(np.floor(U/2)), v + int(np.floor(V/2))] * enlarged[i - u + int(np.floor(U/2)), j - v + int(np.floor(V/2))] # 1 + 0 + 0 + 0 + 

            new_arr[i, j] = sum

    print(f"correct:\n{convolve(arr, k, mode='same')}\n")
    #print(f"ses:\n{convolve(arr, k)}\n")
    print(f"new arr:\n{new_arr}\n")    
    return new_arr

def slow_convolve(arr, k):

    I, J = arr.shape[:2]
    
    new_arr = np.zeros((I, J))
    
    # first = np.flip(k, axis=0)
    # kernel = np.flip(first, axis=1)
    kernel = k[::-1, ::-1] 
    # kernel = k
    U, V = kernel.shape

    lr = U//2
    tb = V//2
    #
    # aufrunden/abrunden aus formel noch richtig implementieren
    #

    # enlarged = np.zeros((I + 2*lr, J + 2*tb))
    # enlarged[lr : lr+I, tb : tb+J] = arr[:, :]

    enlarged = np.pad(arr, ((lr, lr), (tb, tb)), mode="constant", constant_values=0)

    he = U / 2
    lp = V / 2

    for i in range(I):
        for j in range(J):
            sum = 0
            # for u in range(-lr, int(he) + 1 if he > int(he) else int(he)):
            #     for v in range(-tb, int(lp) + 1 if lp > int(lp) else int(lp)):
            #         sum += kernel[u + lr, v + tb] * enlarged[i + u, j + u]
            #
            for u in range(- int(np.floor(U/2)), int(np.ceil(U/2)) - 1):
                for v in range(-int(np.floor(V/2)), int(np.ceil(V/2)) - 1):
                    # sum += k[u + int(np.floor(U/2)), v + int(np.floor(V/2))] * enlarged[i - u, j - v]
                    sum += k[u, v] * enlarged[i + u, j + v]
            new_arr[i, j] = sum
    print(new_arr)
    return new_arr 
"""

if __name__ == '__main__':
    k = make_kernel(9,9/5)   # todo: find better parameters

    # TODO: chose the image you prefer
    im = np.array((Image.open('input1.jpg')).convert("L"))
    # im = np.array((Image.open('input2.jpg')).convert("L"))
    # im = np.array((Image.open('input3.jpg')).convert("L"))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    result = im + (im - slow_convolve(im, k))
    result = np.clip(result, 0, 255).astype(np.uint8)
    bird = Image.fromarray(result)
    bird.save("test.png")

