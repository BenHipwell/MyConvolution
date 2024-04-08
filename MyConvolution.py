import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    # Your code here. You'll need to vectorise your implementation to ensure it runs
    # at a reasonable speed.

    # flips the given kernel in both dimensions to start convolution
    kernel = np.flip(kernel)

    # calculate the necessary padding width, using the value of the dimension which needs the most padding
    # - instead of padding each dimension separately
    pad_width = int((max(kernel.shape) - 1) / 2)
    # create an empty numpy array with the same shape as the given image to store the values of the new pixels
    new_image = np.zeros(image.shape, dtype=np.int64)

    # take appropriate action if the given image is greyscale (2 dimensions)
    if image.ndim == 2:
        # pad the numpy array, using the calculated padding width
        data = np.pad(image, pad_width=pad_width, mode='constant')
        # iterate through each of the pixels in the new padded image
        # however start from the first non padded element (index = pad width)
        # and stop at the last non padded element (index = dimension size + pad width (offset from other side of padding))
        for i in range(pad_width, image.shape[0] + pad_width):
            for j in range(pad_width, image.shape[1] + pad_width):

                # now calculate the index in both dimensions of the 'top left' and 'bottom right' of the window
                # so that I can get all elements within the kernel bounds
                top_left_i = i - int((kernel.shape[0] - 1) / 2)
                top_left_j = j - int((kernel.shape[1] - 1) / 2)

                bottom_right_i = i + int((kernel.shape[0] - 1) / 2)
                bottom_right_j = j + int((kernel.shape[1] - 1) / 2)

                # create the kernel taking the elements in both dimensions between the 'top left' and 'bottom right'
                window = data[top_left_i:bottom_right_i + 1,top_left_j:bottom_right_j + 1]
                # calculates the convoluted result of the window and places it into the correct position of the new image array
                new_image[i-pad_width,j-pad_width] = np.sum(np.multiply(kernel, window))

    # take appropriate action if the given image has separate colour channels (3 dimensions)
    elif image.ndim == 3:
        # pad the numpy array, using the calculated padding width
        data = np.pad(image, ((pad_width,pad_width),(pad_width,pad_width),(0,0)), mode='constant')
        # iterate through each of the pixels in the new padded image
        # however start from the first non padded element (index = pad width)
        # and stop at the last non padded element (index = dimension size + pad width (offset from other side of padding))
        for i in range(pad_width, image.shape[0] + pad_width):
            for j in range(pad_width, image.shape[1] + pad_width):
                # for each pixel, iterate through each colour channel (R G B)
                for k in range(0, image.shape[2]):

                    # now calculate the index in both dimensions of the 'top left' and 'bottom right' of the window
                    # so that I can get all elements within the kernel bounds
                    top_left_i = i - int((kernel.shape[0] - 1) / 2)
                    top_left_j = j - int((kernel.shape[1] - 1) / 2)

                    bottom_right_i = i + int((kernel.shape[0] - 1) / 2)
                    bottom_right_j = j + int((kernel.shape[1] - 1) / 2)

                    # create the kernel taking the elements in both dimensions, and in the correct colour channel, between the 'top left' and 'bottom right'
                    window = data[top_left_i:bottom_right_i + 1,top_left_j:bottom_right_j + 1,k]
                    # calculates the convoluted result of the window and places it into the correct position of the new image array
                    new_image[i - pad_width,j - pad_width,k] = np.sum(np.multiply(kernel, window))

    return new_image