import numpy as np
import cv2
import time
def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global hsv_frame
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        print(str(y)+" "+str(x)+ " = "+ str(hsv_frame[y,x]))
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
time.sleep(3);

for i in range(40):
    _,back = cap.read()

#img = np.zeros((480,400,3), np.uint8)
#print(np.shape(frame))
#img_new = np.zeros(np.shape(frame), np.uint8) red_l = np.array([0,120,70]);red_h = np.array([10,255,255]);red_l = np.array([170,120,70]);red_h = np.array([180,255,255]);
cv2.namedWindow('image')
cv2.setMouseCallback('image',click)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        #frame = cv2.GaussianBlur(frame,(11,11),2)
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);

        #brown_l = np.array([15,30,130]);
        #brown_h = np.array([35,100,240]);
        #deep_blue_l = np.array([100,100,100]);
        #deep_blue_h = np.array([111,255,255]);
        deep_blue_l = np.array([45,70,40]);
        deep_blue_h = np.array([111,255,255]);
        mask_low = cv2.inRange(hsv_frame,deep_blue_l,deep_blue_h);

        #red_l = np.array([340,120,70]);
        #red_h = np.array([359,255,255]);
        #mask_high = cv2.inRange(frame,red_l,red_h);

        mask = mask_low

        mask1 = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));
        mask2 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8));

        mask1 = cv2.bitwise_not(mask2)
        frame_new = cv2.bitwise_and(frame,frame,mask=mask1)


        cloth = cv2.bitwise_and(back,back,mask=mask2)
        #frame_new = cv2.bitwise_and(hsv_frame,hsv_frame,mask = mask)
        
        
        magic_frame = cv2.addWeighted(frame_new,1,cloth,1,0)
        out.write(magic_frame)
        hsv_frame_disp = cv2.cvtColor(magic_frame,cv2.COLOR_BGR2HSV);
        #cv2.imshow('image',mask)
        cv2.imshow('image',magic_frame)
        #cv2.imshow('image2',mask1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else :
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()