from asyncio.windows_events import NULL
from math import atan, ceil, degrees, exp, pi, sqrt
import numpy as np
import cv2 as cv

THRESHOLD_HIGH = 35
THRESHOLD_LOW = 15

####################################################################################
# convolve src image with kernel, return result
# src: image matrix(mxn), numpy.ndarray
def convolve(src, kernel):
    row, col = src.shape

    # Calculate kernel radius for convolution
    radius = int((len(kernel)-1) / 2)

    result = np.zeros((row, col))   #intialize result img with zeros

    # Enumerate over all pixels in src image
    for (centerX, centerY), _ in np.ndenumerate(np.array(src)):
        convolvedValue = 0

        # Convolution for each pixel with kernel
        for u in range(-radius, radius+1):
            for v in range(-radius, radius+1):
                x, y = centerX+u, centerY+v

                # Ignore pixels out of image bounds, treated as zeros (zero-padding)
                if 0 <= x < row and 0 <= y < col:
                    convolvedValue += kernel[radius-u, radius-v] * src[x][y]
        result[centerX, centerY] = convolvedValue

    return result


####################################################################################
# Smooth src image with using gaussian blur
# src: image matrix(mxn), numpy.ndarray
def guassianBlur(src, sigma):
    print("Applying gaussian blur...")

    # Determine size of kernel given sigma
    # Kernel (h, k) where h and k in [-radius, radius] and center at (h,k) = (0,0)
    # Kernel dimensions: hsize * hsize
    radius = ceil(3 * sigma)
    hsize = 2 * radius + 1 

    # Intialize gaussian kernel and populate with Gaussian distribution
    kernel = np.zeros((hsize, hsize)) #intialize kernel with zeros
    kernelSum = 0

    for h in range(-radius, radius+1):
        for k in range(-radius, radius+1):
            twoSigmaSqr = 2 * (sigma ** 2)
            gaussian_hk = exp(-1 * (h**2+k**2) / twoSigmaSqr) / (twoSigmaSqr * pi)

            kernel[radius+h, radius+k] = gaussian_hk
            kernelSum += gaussian_hk
    
    kernel = kernel / kernelSum     #Normalize kernel

    result = convolve(src, kernel)
    return result



####################################################################################
# Smooth src image with using gaussian blur
# img0: image matrix(mxn), numpy.ndarray  
def myEdgeFilter(img0, sigma):
    h, w = img0.shape
    imgGaussian = guassianBlur(img0, sigma)     #Gausssian Blur

    print("Calculating x and y gradients...")
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    imgx = convolve(imgGaussian, sobel_x)   #image gradient in x direction
    imgy = convolve(imgGaussian, sobel_y)   #image gradient in y direction

    # # Calculate gradient magnitude and direction
    print("Calculating image gradient magntude and direction...")
    imgMagnitude = np.zeros(img0.shape)
    imgDirection = np.zeros(img0.shape)
    for (i, j),_ in np.ndenumerate(imgMagnitude):
        imgMagnitude[i,j] = sqrt(imgx[i,j]**2 + imgy[i,j]**2)
        imgDirection[i,j] = degrees(atan(imgy[i,j] / imgx[i,j])) if imgx[i,j] != 0 else 0


    # Non-maximum supression
    print("Applying non-maxima supression...")
    non_maxImg = np.zeros(imgMagnitude.shape)

    for (i, j), magnitude in np.ndenumerate(imgMagnitude):
        if i == 0 or j == 0 or i == h-1 or j == w-1:
            continue
        angle = imgDirection[i,j]
        #Flip the angle if > 180
        if angle > 180:  
            angle -= 180

        # Find the neighboring pixel values given the gradient direction
        neighbor1, neighbor2 = 0, 0

        if 0 <= angle < 22.5 or 157.5 <= angle <= 180:
            neighbor1 = imgMagnitude[i,j-1]
            neighbor2 = imgMagnitude[i,j+1]
        elif 22.5 <= angle < 67.5:
            neighbor1 = imgMagnitude[i-1,j+1]
            neighbor2 = imgMagnitude[i+1,j-1]
        elif 67.5 <= angle < 112.5:
            neighbor1 = imgMagnitude[i-1,j]
            neighbor2 = imgMagnitude[i+1,j]
        else:   #112.5 <= angle < 157.5
            neighbor1 = imgMagnitude[i-1,j-1]
            neighbor2 = imgMagnitude[i+1,j+1]

        # Non-maximum supression
        non_maxImg[i, j] = magnitude if magnitude >= neighbor1 and magnitude >= neighbor2 else 0


    # Double Thresholding with upper and lower thresholds
    img1 = np.zeros(np.shape(non_maxImg))

    for (i, j), magnitude in np.ndenumerate(non_maxImg):
        if magnitude >= THRESHOLD_HIGH:
            img1[i, j] = 255
        elif THRESHOLD_LOW <= magnitude < THRESHOLD_HIGH:
            img1[i, j] = 50

    return imgGaussian, imgx, imgy, imgMagnitude, imgDirection, non_maxImg, img1


####################################################################################
# Main function, displays all resulting images showing the edge detection steps
def main():
    filepath = "cat2.jpg"   #Image file path
    sigma = 1.5               #Sigma for Gaussian filter

    image = cv.imread(filepath)
    img0 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    print("Running Edge Filter on " + filepath + " with sigma: " + str(sigma))
    imgGaussian, imgx, imgy, imgMagnitude, imgDirection, non_maxImg, img1 = myEdgeFilter(img0, sigma)

    images = [("GaussianBlur", imgGaussian), ("Gradient X", imgx), ("Gradient Y", imgy), 
                ("Gradient Magnitude", imgMagnitude), ("Gradient Direction", imgDirection), 
                ("Non-maximum suppression", non_maxImg), ("Canny Edge Detector Result", img1)]

    cv.imshow('Input image', img0)

    for (name, image) in images:
        cv.imwrite('./results/'+ name +'.jpg', image)
        cv.imshow(name, image.astype(np.uint8))
        cv.waitKey(0)
        cv.destroyWindow(name)

if __name__ == "__main__":
    main()

