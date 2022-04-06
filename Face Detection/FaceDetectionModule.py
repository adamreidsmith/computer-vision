import cv2
from mediapipe import solutions as mp_solutions
import time


class faceDetector:
    def __init__(self, model=1, min_detection_confidence=0.5):
        self.model = model
        self.min_detection_confidence = min_detection_confidence

        self.face_detection = mp_solutions.face_detection.FaceDetection(
            model_selection=self.model,
            min_detection_confidence=self.min_detection_confidence
        )
    
    def find_faces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                h, w, c = img.shape
                bb = detection.location_data.relative_bounding_box
                bb_pix = int(bb.xmin * w), int(bb.ymin * h), \
                        int(bb.width * w), int(bb.height * h)
                
                if draw:
                    self._fancy_draw(img, bb_pix)
                    cv2.putText(img, str(int(detection.score[0] * 100)) + '%', (bb_pix[0], bb_pix[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                #if draw:
                #    mp_solutions.drawing_utils.draw_detection(img, detection)

                bboxs.append([id, bb_pix, detection.score])

        return img, bboxs

    def _fancy_draw(self, img, bbox, l=30, t=5, rt=1, color=(255, 0, 255)):
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
        cv2.rectangle(img, bbox, color, rt)
        # Top left
        cv2.line(img, (x0, y0), (x0+l, y0), color, t)
        cv2.line(img, (x0, y0), (x0, y0+l), color, t)
        # Top right
        cv2.line(img, (x1, y0), (x1-l, y0), color, t)
        cv2.line(img, (x1, y0), (x1, y0+l), color, t)
        # Bottom left
        cv2.line(img, (x0, y1), (x0+l, y1), color, t)
        cv2.line(img, (x0, y1), (x0, y1-l), color, t)
        # Bottom right
        cv2.line(img, (x1, y1), (x1-l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1-l), color, t)


        return img

def main():
    cap = cv2.VideoCapture('media/1.mp4')
    max_fps = 30
    t_now = t_prev = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()
        if not success: break

        img, bboxs = detector.find_faces(img)
        
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
