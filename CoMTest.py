#(480,640,3)
import numpy as np
import cv2  # OpenCV module
import time
from tkinter import *
import math

colorID = [0,0,0,0]
COLORS = [(0,255,120),(255,120,120),(0,69,255),(255,255,255)]
POINTS = [(.69154,-.600138),(.67307,-.135855),(.1763488,-.1238),(.13237,-.63746)]
MYPOINTS = [(33,403),(350,400),(370,67),(25,25)]
src = np.array(MYPOINTS)
dst = np.array(POINTS)

x, y, u, v = src[:,0], src[:,1], dst[:,0], dst[:,1]
A = np.zeros((9,9))
j = 0
for i in range(4):
    A[j,:] = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*u[i], y[i]*u[i], u[i]])
    A[j+1,:] = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*v[i], y[i]*v[i], v[i]])
    j += 2
A[8, 8] = 1   # assuming h_9 = 1
b = [0]*8 + [1]

H = np.reshape(np.linalg.solve(A, b), (3,3))
print(H)

class Bottle:
    def __init__(self,clr: list, pos: list):
        self.color = clr
        self.pos = [pos]
        self.time = [time.time_ns()]
        self.interpolate = 0
    def update_pos(self,new_pos: list):
        self.pos.append(new_pos)
        self.time.append(time.time_ns())
    def find_velocity(self):
        self.velX = (self.pos[-1][0]-self.pos[0][0])/(self.time[-1]-self.time[0])
        self.velY = (self.pos[-1][1]-self.pos[0][1])/(self.time[-1]-self.time[0])

def find_contours(i,image,original,colors):
    contoursss, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    min_area = 2500
    contourss = [cnt for cnt in contoursss if (cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < 9000)]
    cv2.drawContours(original, contourss, -1, colors,3)
    if (len(contourss)>0):
        colorID[i] = 1
        for c in contourss:
            # Compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw the contour and center of the shape on the image
            cv2.circle(original, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(original, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print("Pos: ",(cX,cY))
            new = [H[0,0]*cX+H[0,1]*cY+H[0,2],H[1,0]*cX+H[1,1]*cY+H[1,2]]
            print("New Pos: ",new)

def main():
    # Open up the webcam
    cap = cv2.VideoCapture(2)
    while True:

        # Read from the camera frame by frame and crop
        ret, cv_image1 = cap.read()
        cv_image = cv_image1[0:440,138:540]
        #print(len(cv_image))
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # threshold values
        # yellow 
        lower_bound_HSV_yellow = np.array([23, 41, 105]) 
        upper_bound_HSV_yellow = np.array([34, 255, 255])
        lower_bound_HSV_yellow = np.array([21, 39, 103]) 
        upper_bound_HSV_yellow = np.array([36, 255, 255])
        # blue 
        lower_bound_HSV_blue = np.array([71, 135, 68])
        upper_bound_HSV_blue = np.array([105, 255, 255])
        # orange 
        lower_bound_HSV_orange = np.array([15, 100, 255])
        upper_bound_HSV_orange = np.array([23, 255, 255])
        lower_bound_HSV_orange = np.array([13, 90, 200])
        upper_bound_HSV_orange = np.array([25, 255, 255])
        # clear
        # lower_bound_HSV_clear = np.array([255, 255, 255])
        # upper_bound_HSV_clear = np.array([255, 255, 255])
        ret, frame = cap.read()
        cannyIm = cv2.Canny(cv_image, 50, 150, apertureSize = 3)
        cv2.imshow("Canny_Image", cannyIm)
        cv2.waitKey(3)
        time.sleep(0.02)
        
        # create four color masks
        mask_HSV_yellow = cv2.inRange(hsv_image, lower_bound_HSV_yellow, upper_bound_HSV_yellow)
        mask_HSV_blue = cv2.inRange(hsv_image, lower_bound_HSV_blue, upper_bound_HSV_blue)
        mask_HSV_orange = cv2.inRange(hsv_image, lower_bound_HSV_orange, upper_bound_HSV_orange)
        # mask_HSV_clear = cv2.inRange(hsv_image, lower_bound_HSV_clear, upper_bound_HSV_clear)
        
        kernel = np.ones((7,7),np.uint8)
        num_iterations = 3

        ################ Opening ####################
        opening_yellow = cv2.morphologyEx(mask_HSV_yellow, cv2.MORPH_OPEN, kernel, iterations = num_iterations)
        opening_blue = cv2.morphologyEx(mask_HSV_blue, cv2.MORPH_OPEN, kernel, iterations = num_iterations)
        opening_orange = cv2.morphologyEx(mask_HSV_orange, cv2.MORPH_OPEN, kernel, iterations = num_iterations)
        # opening_clear = cv2.morphologyEx(mask_HSV_clear, cv2.MORPH_OPEN, kernel, iterations = num_iterations)
        opening_clear = cv2.morphologyEx(cannyIm, cv2.MORPH_CLOSE, kernel, iterations = num_iterations)

        images = [opening_yellow,opening_blue,opening_orange,opening_clear]
        colorID.clear()
        for img in range(len(images)):
            find_contours(img,images[img],cv_image,COLORS[img])
        if colorID[0]==1:
            print("yellow")
        elif colorID[1]==1:
            print("blue")
        elif colorID[2]==1:
            print("orange")
        elif colorID[3]==1:
            print("clear")

        #cv2.circle(cv_image, (33,403), 7, (255, 255, 255), -1)
        #cv2.circle(cv_image, (350,400), 7, (255, 255, 255), -1)
        #cv2.circle(cv_image, (370,67), 7, (255, 255, 255), -1)
        #cv2.circle(cv_image, (25,25), 7, (255, 255, 255), -1)
        
        ## display image
        cv2.imshow("Original",cv_image)
        cv2.imshow("Opening - Yellow", opening_yellow)
        cv2.imshow("Opening - Blue", opening_blue)
        cv2.imshow("Opening - Orange", opening_orange)
        cv2.imshow("Opening - Clear", opening_clear)
        #print(cv_image.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__=='__main__':
    main()
