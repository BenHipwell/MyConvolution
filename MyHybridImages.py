import math
import numpy as np
import MyConvolution

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray: 
    """     
    Create hybrid images by combining a low-pass and high-pass filtered pair. 

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels)) 
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels)) 
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
    a high-pass image created by subtracting highImage from highImage convolved with 
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input images. 
    :rtype numpy.ndarray
    """

    # creates the low pass filtered image by generating the Gaussian kernel using the given sigma value, and using my solution for convolution
    low = MyConvolution.convolve(lowImage, makeGaussianKernel(lowSigma))
    # creates the high pass filtered image by subtracing a low pass filtered image from the original
    high = np.subtract(highImage, MyConvolution.convolve(highImage, makeGaussianKernel(highSigma)))

    # adding the two images together to create the hybrid image
    result = low + high
    # making sure that all the values are within the correct boundaries of RGB
    np.clip(result, 0, 255, result)

    return result


def makeGaussianKernel(sigma: float) -> np.ndarray: 
    """     
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    # calculates the given size based on the given equation
    size = int(8. * sigma + 1.)
    if size % 2 == 0:
        size += 1

    # calculates the centre by dividing the side length by 2 and flooring it
    centre = size // 2
    # create empty kernel of 0s of the appropriate size
    kernel = np.zeros((size, size))

    # loop through each of the elements in the kernel
    for i in range(size):
        for j in range(size):
            # pass each element in the kernel through the 2D Gaussian function
            kernel[i, j] = np.exp(-((i - centre) ** 2 + (j - centre) ** 2) / (2 * sigma ** 2))

    # make all elements in the kernel sum to 1 by dividing each element by the sum of all elements in the kernel
    kernel = kernel / np.sum(kernel)

    return kernel
