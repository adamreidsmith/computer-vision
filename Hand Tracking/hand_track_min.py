import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2, 1, 0.5, 0.5)
mp_draw = mp.solutions.drawing_utils

t_prev = 0
t_now = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_lmks in results.multi_hand_landmarks:
            for id, lmk in enumerate(hand_lmks.landmark):
                h, w, c = img.shape
                cx, cy = int(lmk.x * w), int(lmk.y * h)
                print(id, cx, cy)

                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_lmks, mp_hands.HAND_CONNECTIONS)

    t_now = time.time()
    fps = 1/(t_now - t_prev)
    t_prev = t_now

    cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)