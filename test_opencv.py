import cv2
import numpy as np

image = cv2.imread("Hasil.jpg", cv2.IMREAD_GRAYSCALE)

# image = cv2.resize(image, (720, 416))

original = image.copy()

# Kernel sharpening
sharpening_kernel = np.array([
    [0, -0.1, 0],
    [-0.1, 1.4, -0.1],
    [0, -0.1, 0]
])

# Menerapkan filter sharpening
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

_, thresholded_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

img = cv2.adaptiveThreshold(sharpened_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)

kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('adaptive',img)
# cv2.imshow('erosion',erosion)
# cv2.imshow('dilate',dilation)
# cv2.imshow('opening',opening)
# cv2.imshow('closing',closing)


cv2.imshow('sharped', thresholded_image)
cv2.imshow('original', original)
cv2.waitKey(0)
cv2.destroyAllWindows()

