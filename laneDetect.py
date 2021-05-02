import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image_functions import *


""" Window Searching """
def laneWindowSearch(bin_image, debug=False):
    nwindows = 10 # number of windows in vertical axis
    
    img_h = bin_image.shape[0] # image height
    histogram = np.sum(bin_image[int(img_h/2):, :], axis=0) # get histogram
    
    midpoint = np.int(histogram.shape[0]/2) # midpoint of histogram chart
    leftx_base = np.argmax(histogram[:midpoint]) # starting point of left lane
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # starting point of right lane
    
    window_height = np.int(img_h/nwindows) # height of each window
    
    # get nonzero pixels position in x and y axis
    nonzero = bin_image.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    
    # set current index of left and right window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 70 # margin of a window
    minpix = 20 # minimum active pixels in a window
    
    # left and right lane indices
    left_lane_inds = []
    right_lane_inds = []
    
    # window for debugging
    debug_sliding_win = np.dstack((bin_image, bin_image, bin_image))
    debug_sliding_win[:,:,0] = bin_image[:,:]
    
    for window in range(nwindows):
        # window margin
        win_y_low = img_h - (window+1)*window_height
        win_y_high = img_h - (window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # draw rectangle on debug window
        if leftx_base != 0: # no left lane
            cv.rectangle(debug_sliding_win, (win_xleft_low, win_y_high), (win_xleft_high, win_y_low), BGR_GREEN, 2)
        if rightx_base != 250: # no right lane
            cv.rectangle(debug_sliding_win, (win_xright_low, win_y_high), (win_xright_high, win_y_low), BGR_RED, 2)
#
        # check for any active pixels in the current window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # extract left and right lane
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # apply polyfit
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [[0]]
    
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [[0]]
    
    #draw output
    ploty = np.linspace(0, img_h - 1, img_h)
    try:
        left_fitx = (left_fit[0] * ploty**2) + (left_fit[1]*ploty) + left_fit[2]
        cv.polylines(debug_sliding_win, np.int32([np.column_stack((left_fitx, ploty))]), False, (255, 255, 0), 3)
    except:
        left_fitx = [[0]]
    
    try:
        right_fitx = (right_fit[0] * ploty**2) + (right_fit[1]*ploty) + right_fit[2]
        cv.polylines(debug_sliding_win, np.int32([np.column_stack((right_fitx, ploty))]), False, (255, 255, 0), 3)
    except:
        right_fitx = [[0]]

    # if left_fitx is not [[0]]:
    #     cv.polylines(debug_sliding_win, np.int32([np.column_stack((left_fitx, ploty))]), False, (255, 255, 0), 3)

    # if right_fitx is not [[0]]:
    #     cv.polylines(debug_sliding_win, np.int32([np.column_stack((right_fitx, ploty))]), False, (255, 255, 0), 3)
    
    # plotImage(debug_sliding_win, "Polyfit implimentation")
    
    return left_fit, right_fit, debug_sliding_win.astype("uint8")


def findLaneRecursive(warped, left_fit, right_fit, margin=50, return_img=False, plot_boxes=False, plot_line=False):

    """
    This is the recursive version of the lane finder. It needs information from the previous frame in order to
    locate the lane lines.

    Margin controls how wide from the previous lane line equation we are going to look

    """

    nonzero = warped.nonzero() # get nonzero pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Get left fit and right fit
    left_lane_inds  = ((nonzerox > ((left_fit[0]*(nonzeroy**2)) + (left_fit[1]*nonzeroy) + left_fit[2] - margin)) & 
                       (nonzerox < ((left_fit[0]*(nonzeroy**2)) + (left_fit[1]*nonzeroy) + left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > ((right_fit[0]*(nonzeroy**2)) + (right_fit[1]*nonzeroy) + right_fit[2] - margin)) & 
                       (nonzerox < ((right_fit[0]*(nonzeroy**2)) + (right_fit[1]*nonzeroy) + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [np.array([0])]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [np.array([0])]

    if return_img:
        img_h = warped.shape[0]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_h - 1, img_h)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        if plot_line:
            if left_fit != [[0]]:
                cv.polylines(out_img, np.int32([np.column_stack((left_fitx, ploty))]), False, (255, 255, 0), 2)
            
            if right_fit != [[0]]:
                cv.polylines(out_img, np.int32([np.column_stack((right_fitx, ploty))]), False, (255, 255, 0), 2)

        if plot_boxes:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            if left_fit != [[0]]:
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
            
            if right_fit != [[0]]:
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)
            out_img = result.copy()

        return left_fit, right_fit, out_img

    return left_fit, right_fit


def getLaneOffset(image, left_fit, right_fit, imsize=(720, 1280), x_mpp=3.7 / 1180):
    warp_ratio = 2.56

    # img_h is the starting warped point
    img_h = 400 #imsize[0]
    img_center = imsize[1] / 2

    custom_font = cv.FONT_HERSHEY_SIMPLEX

    # fine the beginning point of the lane
    if left_fit[0] != [0]:
        left_lane_start = left_fit[0]*(img_h**2) + left_fit[1]*img_h + left_fit[2]
    else:
        left_lane_start = None
    
    if right_fit[0] != [0]:
        right_lane_start = right_fit[0]*(img_h**2) + right_fit[1]*img_h + right_fit[2]
    else:
        right_lane_start = None

    if left_lane_start is not None and right_lane_start is not None:
        lane_center = np.mean((left_lane_start, right_lane_start)) * warp_ratio # multiply by ratio
        dist = (img_center - lane_center) * x_mpp * warp_ratio
    else:
        lane_center = None
        dist = None


    lane_keep_status = ""
    custom_color = (BGR_RED)

    if dist == None:
        side = 'none'
        lane_keep_status = "NO LANE DETECED"
    elif dist > 0:
        side = 'left'
    else:
        side = 'right'

    # lane keeping status
    if dist != None:
        abs_dist = abs(dist)
        if abs_dist >= 1.85:
            lane_keep_status = "BAD LANE DETECT"
        elif abs_dist >= 0.85:
            lane_keep_status = "LEAVING LANE"
        elif abs_dist >= 0.5:
            lane_keep_status = "TOO CLOSE ON A SIDE"
            custom_color = (51, 153, 255) # orange
        else:
            lane_keep_status = "GOOD LANE-KEEPING"
            custom_color = (51, 204, 51) # green

    
    img_h = 720
    # Draw Car Center
    cv.line(image, (int(img_center), img_h - 20), (int(img_center), int(img_h) - 100), BGR_RED, 2) # draw center of car
    cv.putText(image, "Car Center", (int(img_center) -150, img_h - 50), custom_font, 0.8, BGR_RED, 2)
    
    # Draw Line Center
    if dist != None:
        cv.line(image, (int(lane_center), img_h - 20), (int(lane_center), int(img_h) - 80), BGR_BLUE, 2) # draw lane center
        cv.putText(image, "Lane Center", (int(lane_center) + 10, img_h - 30), custom_font, 0.8, BGR_BLUE, 2)
    
        # Draw Distance from center Line
        cv.line(image, (int(lane_center), img_h - 40), (int(img_center), img_h - 40), (200, 250, 150), 2)

    # Draw status
    cv.putText(image, lane_keep_status, (800, 50), custom_font, 1.3, custom_color, 2)

    if dist != None:
        cv.putText(image, "Distance from center: " + str(abs(round(dist, 2))) +"m", (800, 90), custom_font, 0.8, custom_color, 2)
        cv.putText(image, "Suggest Steering: " + side, (800, 120), custom_font, 0.8, custom_color, 2)
        dist = np.absolute(dist)
    

    return dist, side
    
# inv_warped = cv.warpPerspective(debug_sliding_win, Minv, (1280, 720))
# plotImage(inv_warped)
    
