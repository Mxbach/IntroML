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

    enlarged = np.pad(arr, ((tb, tb), (lr, lr)), mode="constant", constant_values=0)

    he = U / 2
    lp = V / 2

    for i in range(I):
        for j in range(J):
            sum = 0
            # for u in range(-lr, int(he) + 1 if he > int(he) else int(he)):
            #     for v in range(-tb, int(lp) + 1 if lp > int(lp) else int(lp)):
            #         sum += kernel[u + lr, v + tb] * enlarged[i + u, j + u]
            #
            for u in range(- int(np.floor(U/2)), int(np.ceil(U/2))):
                for v in range(-int(np.floor(V/2)), int(np.ceil(V/2))):
                    sum += k[u + int(np.floor(U/2)), v + int(np.floor(V/2))] * enlarged[i - u, j - v]
            new_arr[i, j] = sum
    print(new_arr)
    return new_arr 


if __name__ == '__main__':
    k = make_kernel(9,9/5)   # todo: find better parameters
    # print(k)
    # TODO: chose the image you prefer
    im = np.array((Image.open('input1.jpg')).convert("L"))
    # bw.save("bird_black_white.png")
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    result = im + (im - slow_convolve(im, k))

    bird = Image.fromarray(result)
    bird.convert("RGB")
    bird.save("test.png")
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
