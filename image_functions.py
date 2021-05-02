"""
This module is used for image processing including Camera Calibration, Image
Filtering, Color Space Alteration and more.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

""" COLOR CONSTANTS """
BGR_RED   = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE  = (255, 0, 0)
BGR_BLACK = (0, 0, 0)
BGR_WHITE = (255, 255, 255)

""" SWITCHES """



def BGR2RGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def plotImage(image, title_name="", isBinary=False):
    """
    Plot image in Matplotlib Figure Window
    :param title_name: title of the image
    :param image: image to be plotted
    :return: None
    """    
    if not isBinary:
        try: # if image is in BGR
            image = BGR2RGB(image)  # convert image from BGR to RGB
        except: # if image is already in RGB
            pass;

    plt.figure(figsize=(10,5))
    plt.title(title_name)
    plt.imshow(image)
    plt.show()

def perspectiveTransform(image, output_size=500, debug=False):
    # source points

    # last best resources -> Good with curves
    # src_topleft  = (480, 431)
    # src_topright = (800, 431)
    # src_botleft  = (50, 600)
    # src_botright = (1230, 600)

    # current use -> Good with straight
    src_topleft  = (480, 400)
    src_topright = (800, 400)
    src_botleft  = (50, 600)
    src_botright = (1230, 600)

    cv.circle(image, src_topleft, 1, BGR_BLUE)
    cv.circle(image, src_topright, 1, BGR_GREEN)
    cv.circle(image, src_botright, 1, BGR_RED)
    cv.circle(image, src_botleft, 1, BGR_WHITE)

    # Determine the output size of the transformed image
    dest_topleft  = (0, 0)
    dest_topright = (output_size, 0)
    dest_botleft  = (0, output_size)
    dest_botright = (output_size, output_size)

    # used for showing perspective points
    if debug:
        debug_img = np.copy(image)
        cv.circle(debug_img, src_topleft, 5, BGR_RED, -1)
        cv.circle(debug_img, src_topright, 5, BGR_GREEN, -1)
        cv.circle(debug_img, src_botleft, 5, BGR_BLUE, -1)
        cv.circle(debug_img, src_botright, 5, BGR_BLACK, -1)
        plotImage(debug_img, "Debug Perspective")

    src = np.float32([src_topleft, src_topright, src_botleft, src_botright])
    dest = np.float32([dest_topleft, dest_topright, dest_botleft, dest_botright])

    M = cv.getPerspectiveTransform(src, dest)
    Minv = cv.getPerspectiveTransform(dest, src)
    warped = cv.warpPerspective(image, M, (output_size, output_size))

    return warped, M, Minv

def hls_thresh(image, thresh_min=200, thresh_max=255):
    # convert RGB to HLS and extract the L Channel
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]

    # create image masked in S Channel
    l_bin = np.zeros_like(l_channel)
    l_bin[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1

    return l_bin

def sobel_thresh(image, sobel_kernel=3, orient='x', thresh_min=20, thresh_max=100):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    if orient == 'x':
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    else:
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    grad_bin = np.zeros_like(scaled_sobel)
    grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return grad_bin

def lab_b_thresh(image, thresh=(190, 255)):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lab_b = lab[:,:,2]

    if np.max(lab_b) > 175:
        lab_b = lab_b * (255/np.max(lab_b))

    lab_b_bin = np.zeros_like(lab_b)
    lab_b_bin[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1

    return lab_b_bin

def process_threshold(image):
    dim = (image.shape[1], image.shape[1])
    
    hls = hls_thresh(image, 170) 
    lab_b = lab_b_thresh(image)

    out = np.zeros(dim)
    out[(hls == 1) | (lab_b == 1)] = 1

    return out

def mask_lane_center(image, use_polygon=True):
    if use_polygon:
        pt1 = (300, 720)
        pt2 = (560, 500)
        pt3 = (720, 500)
        pt4 = (980, 720)
        pts = np.array([pt1, pt2, pt3, pt4])

    else: # use triangle instead
        pt1 = (300, 720)
        pt2 = (640, 400)
        pt3 = (980, 720)
        pts = np.array([pt1, pt2, pt3])
    
    res = cv.drawContours(image, [pts], 0, (0,0,0), -1)

    return res