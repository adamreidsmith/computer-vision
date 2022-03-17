import cv2
from mediapipe import solutions as mp_solutions
import time

class handDetector:
    def __init__(self, 
                 mode=False, 
                 max_num_hands=2, 
                 complexity=1, 
                 detection_confidence=0.5, 
                 track_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.hands = mp_solutions.hands.Hands(self.mode, 
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_confidence,
                                         self.track_confidence)
        self.mp_draw = mp_solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_lmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lmks, mp_solutions.hands.HAND_CONNECTIONS)
        
        return img

    def find_position(self, img, hand_no=0):
        lmk_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lmk in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lmk.x * w), int(lmk.y * h)
                lmk_list.append([id, cx, cy])
        
        return lmk_list


def main():
    cap = cv2.VideoCapture(0)

    t_prev = t_now = 0

    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success: break
        img = detector.find_hands(img)
        lmk_list = detector.find_position(img)
        if len(lmk_list) != 0:
            print(lmk_list[0])

        t_now = time.time()
        fps = 1/(t_now - t_prev)
        t_prev = t_now

        cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()