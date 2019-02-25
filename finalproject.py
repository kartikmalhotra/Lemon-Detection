import cv2
import numpy as np

#Define HSV range of yellow color 
HSV_Lower=np.array([25,100,100])
HSV_Upper=np.array([65,255,255])

#Input sample video
cam=cv2.VideoCapture('/Users/hp/Pictures/Camera Roll/karfinal3.mp4')

#Two kernels to perform morphological operation on video frames( i.e., MASKING)
window1=np.ones((5,5))
window2=np.ones((20,20))

temp=0

#Loop till we retrieve frames from inputted video
while True:
    #Retrieve frames of the input video
    ret,frame=cam.read()
    #Resize the frame
    frame=cv2.resize(frame,(700,700))
    #Convert BGR to HSV
    imgHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Masking those objects in the HSV image which are in HSV range defined above(in this case only yellow objects i.e., LEMON)
    img_mask1=cv2.inRange(imgHSV,HSV_Lower,HSV_Upper)
    #Removing noise surrounding the object tracked
    img_mask2=cv2.morphologyEx(img_mask1,cv2.MORPH_OPEN,window1)
    #Removing noise within the object tracked
    img_mask3=cv2.morphologyEx(img_mask2,cv2.MORPH_CLOSE,window2)
    img_mask4=img_mask3

    #Retrieve number of pixels of finally masked image(img_mask3) whose value is non-zero(i.e., to check whether img_mask3 is black)
    if  cv2.countNonZero(img_mask3) == 0 :
            print("****LEMON FALLEN*****")
            break
    else:
        if temp==0 :
            print("****LEMON IS ON THE SPOON*****")
        temp=temp+1
        
    #To get contours or boundaries of the tracked objects 
    _,contours,hierarchy=cv2.findContours(img_mask4.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

     #To bound the tracked object(i.e., lemon) in a rectangle
    for i in range(len(contours)):

        x,y,w,h=cv2.boundingRect(contours[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # To show the masked images
    cv2.imshow("img_mask3",img_mask3)
    cv2.imshow("img_mask2",img_mask2)
    cv2.imshow("img_mask1",img_mask1)
    cv2.imshow("cam",frame)
    cv2.waitKey(10)

                                                                   #************FINISH******************
