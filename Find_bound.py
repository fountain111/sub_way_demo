import cv2
import numpy as np

#img_dir = "dir"

#im = cv2.imread()

def get_bound_rects(im):

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray,(5,5),0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rects contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    return rects

