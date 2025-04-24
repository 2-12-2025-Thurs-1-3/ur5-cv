# 2.12 Lab 7 object detection: a node for observing erosion/dilation
# Jacob Guggenheim 2019
# Jerry Ng 2019, 2020

import numpy as np
import cv2  # OpenCV module
import time
from tkinter import *

import math

# tk = Tk()
# l_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, lower', orient = HORIZONTAL)
# l_h.pack()
# u_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, upper', orient = HORIZONTAL)
# u_h.pack()
# u_h.set(255)
# l_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, lower', orient = HORIZONTAL)
# l_s.pack()
# u_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, upper', orient = HORIZONTAL)
# u_s.pack()
# u_s.set(255)
# l_v = Scale(tk, from_ = 0, to = 255, label = 'Value, lower', orient = HORIZONTAL)
# l_v.pack()
# u_v = Scale(tk, from_ = 0, to = 255, label = 'Value, upper', orient = HORIZONTAL)
# u_v.pack()
# u_v.set(255)

def main():
    # Open up the webcam
    cap = cv2.VideoCapture(2)
    while True:
        #tk.update()

        # Read from the camera frame by frame
        ret, cv_image = cap.read()
        # visualize it in a cv window
        #  cv2.imshow("Original_Image", cv_image)
        #cv2.waitKey(3)

        ################ HSV THRESHOLDING ####################
        # convert to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # get threshold values
        # lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
        # upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])
        lower_bound_yellow = np.array([23,41,105])
        upper_bound_yellow = np.array([34,255,255])

        # threshold
        mask_HSV = cv2.inRange(hsv_image, lower_bound_yellow, upper_bound_yellow)

        # display image
        #  cv2.imshow("HSV_Thresholding", mask_HSV)
        #cv2.waitKey(3)

        # kernel for all morphological operations
        #TOD: Change size of kernel
        # Also, try changing the shape of the kernel (places 1's in certain locations). Try making a circle/line/etc.
        kernel = np.ones((7,7),np.uint8)

        # EXAMPLE OF A VERTICAL LINE:
        #kernel = np.array([[0, 1, 0, 0],\
        #                [0, 1, 0, 0 ],\
        #                [1, 1, 1, 1 ],\
        #                [0, 1, 0, 0]], dtype=np.uint8)

        #TOD: Change number of iterations to see the effect.
        num_iterations = 5
        ################ Opening ####################
        # erode blobs
        test = mask_HSV.copy()
        erosion = cv2.erode(test,kernel, iterations = num_iterations)
        openIm = cv2.dilate(erosion,kernel, iterations=num_iterations)
        test=openIm.copy()

        #ret, thresh = cv2.threshold(test, 150, 255, cv2.THRESH_BINARY)
        contoursss, _ = cv2.findContours(openIm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        min_area = 2500
        contourss = [cnt for cnt in contoursss if (cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < 6000)]
        cv2.drawContours(cv_image, contourss, -1, (0, 255, 0),3)
        for c in contourss:
            # Compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw the contour and center of the shape on the image
            cv2.drawContours(cv_image, [c], -1, (0, 255, 0), 2)
            cv2.circle(cv_image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(cv_image, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #mass_x, mass_y = np.where(erosion>=255)
        #cent_x = np.average(mass_x)
        #cent_y = np.average(mass_y)
        ##cv2.drawContours(erosion)
        #print(cent_x, cent_y)

        ## display image
        cv2.imshow("trhres",cv_image)
        cv2.imshow("Erosion", openIm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__=='__main__':
    main()
