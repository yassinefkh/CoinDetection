import cv2 as cv

def apply_laplace(gray):
    ddepth = cv.CV_16S
    kernel_size = 3
    gray = cv.convertScaleAbs(cv.Laplacian(gray, ddepth, ksize=kernel_size))
    return gray

def apply_erosion(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img = cv.erode(img, kernel, iterations=1)
    return img

def apply_opening(img, iterations=1):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations)