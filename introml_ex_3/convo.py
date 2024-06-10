from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = (ksize//2, ksize//2)
    for i in range(ksize):
        for j in range(ksize):
            x = np.abs(i - center[0])
            y = np.abs(j - center[1])
            kernel[i, j] = 1/(2*np.pi* sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    return kernel / np.sum(kernel)  # implement the Gaussian kernel here


def slow_convolve(arr, k):
    I, J = arr.shape[:2]
    U, V = k.shape
    new_arr = np.zeros((I, J))
    lr = U//2
    tb = V//2
    
    #
    # aufrunden/abrunden aus formel noch richtig implementieren
    #

    enlarged = np.zeros((I + 2*lr, J + 2*tb))
    enlarged[lr : lr+I, tb : tb+J] = arr[:, :]

    for i in range(I):
        for j in range(J):
            sum = 0
            for u in range(-lr, lr):
                for v in range(-tb, tb):
                    sum += k[u + lr, v + tb] * enlarged[i - u, j - v]
            
            new_arr[i, j] = sum

    return new_arr # implement the convolution with padding here


if __name__ == '__main__':
    k = make_kernel(3,3)   # todo: find better parameters
    # print(k)
    # TODO: chose the image you prefer
    im = np.array((Image.open('input1.jpg')))
    bw = im.convert("L")
    # bw.save("bird_black_white.png")
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    result = bw + (bw - slow_convolve(bw, k))
    bird = Image.fromarray(result)
    bird = bird.convert("L")
    bird.save("test.png")
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
