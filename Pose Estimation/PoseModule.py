import cv2
from mediapipe import solutions as mp_solutions
import time

class poseDetector:
    def __init__(self,
                 mode=False,
                 complexity=1,
                 smooth_landmarks=True,
                 enable_segemntation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segemntation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.pose = mp_solutions.pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp_solutions.drawing_utils

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, mp_solutions.pose.POSE_CONNECTIONS)

        return img
    
    def find_position(self, img, draw=True):
        lmk_list = []

        if self.results.pose_landmarks:
            for id, lmk in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lmk.x * w), int(lmk.y * h)

                lmk_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmk_list



def main():
    cap = cv2.VideoCapture('media/6.mp4')
    max_fps = 30
    t_now = t_prev = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        lmk_list = detector.find_position(img)
#         print(lmk_list)

        elapsed = time.time() - t_prev
        if 1/elapsed > max_fps:
            time.sleep(1/max_fps - elapsed)
            
        t_now = time.time()
        fps = 1/(t_now - t_prev)
        t_prev = t_now

        cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
