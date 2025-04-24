#(480,640,3)
import numpy as np
import cv2  # OpenCV module
import time
from tkinter import *
import math

def main():
    # Open up the webcam
    cap = cv2.VideoCapture(2)
    while True:
        #tk.update()

        # Read from the camera frame by frame
        ret, cv_image1 = cap.read()
        cv_image = cv_image1[0:440,138:540]
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # get threshold values
        # lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
        # upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])
        lower_bound_yellow = np.array([23,41,105])
        upper_bound_yellow = np.array([34,255,255])
        mask_HSV = cv2.inRange(hsv_image, lower_bound_yellow, upper_bound_yellow)
        kernel = np.ones((7,7),np.uint8)

        num_iterations = 5
        ################ Opening ####################
        # erode blobs
        test = mask_HSV.copy()
        #erosion = cv2.erode(test,kernel, iterations = num_iterations)
        #openIm = cv2.dilate(erosion,kernel, iterations=num_iterations)
        #test=openIm.copy()
        openIm = cv2.morphologyEx(test, cv2.MORPH_OPEN, kernel)

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

        ## display image
        cv2.imshow("Original",cv_image)
        cv2.imshow("Opening", openIm)
        print(cv_image.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__=='__main__':
    main()
