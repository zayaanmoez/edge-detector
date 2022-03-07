from asyncio.windows_events import NULL
from math import atan, ceil, degrees, exp, pi, sqrt
import numpy as np
import cv2 as cv

THRESHOLD_HIGH = 100
THRESHOLD_LOW = 50

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
    
    # Apply sticks Filter
    imageFiltered = sticks_filter(imgMagnitude.astype(np.uint8), 5)

    # Non-maximum supression
    print("Applying non-maxima supression...")
    non_maxImg = np.zeros(imageFiltered.shape)

    for (i, j), magnitude in np.ndenumerate(imageFiltered):
        if i == 0 or j == 0 or i == h-1 or j == w-1:
            continue
        angle = imgDirection[i,j]
        #Flip the angle if > 180
        if angle > 180:  
            angle -= 180

        # Find the neighboring pixel values given the gradient direction
        neighbor1, neighbor2 = 0, 0

        if 0 <= angle < 22.5 or 157.5 <= angle <= 180:
            neighbor1 = imageFiltered[i,j-1]
            neighbor2 = imageFiltered[i,j+1]
        elif 22.5 <= angle < 67.5:
            neighbor1 = imageFiltered[i-1,j+1]
            neighbor2 = imageFiltered[i+1,j-1]
        elif 67.5 <= angle < 112.5:
            neighbor1 = imageFiltered[i-1,j]
            neighbor2 = imageFiltered[i+1,j]
        else:   #112.5 <= angle < 157.5
            neighbor1 = imageFiltered[i-1,j-1]
            neighbor2 = imageFiltered[i+1,j+1]

        # Non-maximum supression
        non_maxImg[i, j] = magnitude if magnitude >= neighbor1 and magnitude >= neighbor2 else 0


    # Double Thresholding with upper and lower thresholds

    img1 = np.zeros(np.shape(non_maxImg))

    for (i, j), magnitude in np.ndenumerate(non_maxImg):
        if magnitude >= THRESHOLD_HIGH:
            img1[i, j] = 255
        elif THRESHOLD_LOW <= magnitude < THRESHOLD_HIGH:
            img1[i, j] = 50

    return imgMagnitude, imageFiltered, non_maxImg, img1
    

####################################################################################
####################################################################################
####################################################################################
# STICKS FILTER
# Build kernels for Sticks Filter given n
# n: Length of stick features, , must be odd
def getSticksKernels(n):
    i = 2*n - 2
    print("Building stick kernels of length " + str(n) + "...")

    sticks_kernels = []
    #i possible orientations, angle difference between orientations
    orientation = 180 / i
    stick_value = 1 / n
    centerKernel = int((n-1)/2)

    # Build stick kernels based on orientation and stick length
    for ki in range(i):
        angle = ki * orientation
        kernel = np.zeros((n, n), dtype=float)
        if angle == 0:
            for idx in range(n):
                kernel[centerKernel, idx] = stick_value
        elif angle == 45:
            for idx in range(n):
                kernel[idx, n-1-idx] = stick_value
        elif angle == 135:
            for idx in range(n):
                kernel[idx, idx] = stick_value
        elif angle == 90:
            for idx in range(n):
                kernel[idx, centerKernel] = stick_value
        else:
            # Finding the starting edge pixel of the orientation given angle
            pixelr = int((n-1)/2)    #Right edge center pixel row
            pixelc =  n-1    #Right edge center pixel col
            for idx in range(ki):
                if pixelc == n-1 and pixelr > 0:
                    pixelr -= 1
                elif pixelr == 0 and pixelc > 0:
                    pixelc -= 1
                else:
                    pixelr += 1
            move = centerKernel     #Steps to move before switching columns or rows
            count = 0
            if pixelr == 0:    #Sticks starting top of the kernel
                sign = -1 if pixelc > centerKernel else 1   #Vertical direction of movement
                for idx in range(n):
                    if idx == centerKernel:
                        pixelc += sign
                        count = move
                    elif count == move:
                        pixelc += sign
                        count = 0
                    else:
                        count += 1
                    kernel[pixelr+idx, pixelc] = stick_value
            else:   #Sticks starting sides of the kernel
                signH = -1 if pixelc == n-1 else 1      #Horizontal direction of movement           
                for idx in range(n):
                    if idx == centerKernel:
                        pixelr += 1
                        count = move
                    elif count == move:
                        pixelr += 1
                        count = 0
                    else:
                        count += 1
                    kernel[pixelr, pixelc+(signH*idx)] = stick_value
            
            # Create sticks
        sticks_kernels.append(kernel)
    return sticks_kernels


# Sticks Filter
# https://gigl.scs.carleton.ca/sites/default/files/david_mould/bnw-stick-2015.pdf
# src: image matrix(mxn), numpy.ndarray
# n: Length of stick features, must be odd
def sticks_filter(src, n):
    kernels = getSticksKernels(n)

    print("Applying the sticks filter...")
    i = 2*n - 2
    si = []

    for kernel in kernels:
        img = cv.filter2D(src, -1, cv.flip(kernel, -1), borderType=cv.BORDER_CONSTANT)
        si.append(img)

    #image = cv.copyMakeBorder(src, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=0)

    averageIntensity = cv.filter2D(src, -1, np.ones((n, n)), borderType=cv.BORDER_CONSTANT)

    imgFiltered = np.zeros(src.shape)

    for (x, y), brightness in np.ndenumerate(src):
        deltas = []
        avgNeigborI = averageIntensity[x, y] / n**n   #Average intensity of neighboring pixels
        for s in range(0, i):
            deltas.append(abs(float(si[s][x, y]) - float(avgNeigborI)))
        t_xy = max(deltas)
        sign = 1 if brightness >= averageIntensity[x, y] else -1
        imgFiltered[x, y] = brightness + (t_xy)

    return imgFiltered



####################################################################################
# Main function, displays all resulting images showing the edge detection steps
def main():
    filepath = "cat2.jpg"   #Image file path
    sigma = 1.5               #Sigma for Gaussian filter

    image = cv.imread(filepath)
    img0 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    print("Running Edge Filter with Sticks Filter on " + filepath + " with sigma: " + str(sigma))
    imgMagnitude, filteredImg, non_maxImg, img1 = myEdgeFilter(img0, sigma)

    images = [("Gradient Magnitude", imgMagnitude), ("Sticks Filter Output", filteredImg), 
                ("Non-maximum suppression + Sticks", non_maxImg), ("Canny + Sticks Filter Result", img1)]

    cv.imshow('Input image', img0)

    for (name, image) in images:
        cv.imwrite('./results/' + name +'.jpg', image)
        cv.imshow(name, image.astype(np.uint8))
        cv.waitKey(0)
        cv.destroyWindow(name)

if __name__ == "__main__":
    main()

