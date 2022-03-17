import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('media/2.mp4')

t_now = t_prev = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lmk in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape

            cx, cy = int(lmk.x * w), int(lmk.y * h)

            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


    t_now = time.time()
    fps = 1/(t_now - t_prev)
    t_prev = t_now

    cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)