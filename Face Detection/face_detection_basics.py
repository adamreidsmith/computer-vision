import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('media/5.mp4')

t_now = t_prev = 0

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success: break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mp_draw.draw_detection(img, detection)

            h, w, c = img.shape
            bb = detection.location_data.relative_bounding_box
            bb_pix = int(bb.xmin * w), int(bb.ymin * h), \
                     int(bb.width * w), int(bb.height * h)
            
            cv2.rectangle(img, bb_pix, (255, 0, 255), 2)
            cv2.putText(img, str(int(detection.score[0] * 100)) + '%', (bb_pix[0], bb_pix[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    t_now = time.time()
    fps = 1/(t_now - t_prev)
    t_prev = t_now

    cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)