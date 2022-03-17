import cv2
import time
import numpy as np
import HandTrackingModule as htm
from osascript import osascript

w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
t_prev = 0

detector = htm.handDetector(detection_confidence=0.8)

vol = osascript('get volume settings')[1]
vol = int(vol[14:17]) if vol[17] == ',' else (int(vol[14]) if vol[15] == ',' else int(vol[14:16]))

while True:
    success, img = cap.read()

    img = detector.find_hands(img)

    lmks = detector.find_position(img)
    if lmks:
        thumb, index = np.array(lmks[4][1:]), np.array(lmks[8][1:])
        center = (thumb + index)//2

        cv2.circle(img, thumb, 10, (255,0,0), cv2.FILLED)
        cv2.circle(img, index, 10, (255,0,0), cv2.FILLED)
        cv2.line(img, thumb, index, (255,0,0), 3)

        length = np.linalg.norm(thumb - index)

        cv2.circle(img, center, 10, (255,0,0), cv2.FILLED)
        if length < 50:
            cv2.circle(img, center, 10, (0,255,0), cv2.FILLED)
        
        min_vol, max_vol = 0, 100
        vol = np.interp(length, (50, 280), (min_vol, max_vol))
        
        osascript(f'set volume output volume {vol}')
        
    vol_bar = np.interp(vol, (0, 100), (450, 150))
    cv2.rectangle(img, (30,150), (60,450), (0,255,0), 3)
    cv2.rectangle(img, (30,int(vol_bar)), (60,450), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (65,448), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    t_now = time.time()
    fps = 1/(t_now - t_prev)
    t_prev = t_now
    cv2.putText(img, str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)
