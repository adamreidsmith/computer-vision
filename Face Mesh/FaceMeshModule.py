import cv2
from mediapipe import solutions as mp_solutions
import time

class faceMeshDetector:
    def __init__(self,
                 mode=False,
                 max_num_faces=2,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp_solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = mp_solutions.face_mesh.FaceMesh(
            self.mode,
            self.max_num_faces,
            self.refine_landmarks,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
    
    def find_face_mesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lmks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        face_lmks, 
                        mp_solutions.face_mesh.FACEMESH_CONTOURS,
                        self.draw_spec,
                        self.draw_spec
                    )

                face = []
                for id, lmk in enumerate(face_lmks.landmark):
                    h, w, c = img.shape
                    x, y = int(lmk.x * w), int(lmk.y * h)

                    face.append([id, x, y])
                faces.append(face)
        
        return img, faces

def main():
    cap = cv2.VideoCapture('media/5.mp4')
    t_prev = 0
    detector = faceMeshDetector()
    while True:
        success, img = cap.read()
        if not success: break

        img, faces = detector.find_face_mesh(img)

        t_now = time.time()
        fps = 1/(t_now - t_prev)
        t_prev = t_now

        cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

        cv2.imshow('image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()