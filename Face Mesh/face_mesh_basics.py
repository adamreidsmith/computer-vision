import cv2
from mediapipe import solutions as mp_solutions
import time

cap = cv2.VideoCapture('media/5.mp4')

t_prev = 0

mp_draw = mp_solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    success, img = cap.read()
    if not success: break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)
    if results.multi_face_landmarks:
        for face_lmks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                img, 
                face_lmks, 
                mp_solutions.face_mesh.FACEMESH_CONTOURS,
                draw_spec,
                draw_spec
            )
            for id, lmk in enumerate(face_lmks.landmark):
                h, w, c = img.shape
                x, y = int(lmk.x * w), int(lmk.y * h)
                print(id, x, y)

    t_now = time.time()
    fps = 1/(t_now - t_prev)
    t_prev = t_now

    cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)