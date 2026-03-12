import cv2
import numpy as np
import logging
import mediapipe as mp
from scipy.spatial import distance
from config import Config

logger = logging.getLogger(__name__)

class DrowsinessPredictor:
    # Key Landmarks
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
    NOSE_TIP = 1
    LEFT_EAR_POINT = 234
    RIGHT_EAR_POINT = 454

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        logger.info("MediaPipe FaceMesh Initialized - Elite Mode Active.")

    def _landmarks_to_np(self, landmarks, w, h):
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    def _ear(self, eye_points):
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def _mar(self, mouth_points):
        A = distance.euclidean(mouth_points[1], mouth_points[7])
        B = distance.euclidean(mouth_points[2], mouth_points[6])
        C = distance.euclidean(mouth_points[3], mouth_points[5])
        D = distance.euclidean(mouth_points[0], mouth_points[4])
        return (A + B + C) / (3.0 * D)

    def _distraction(self, landmarks):
        nose = landmarks[self.NOSE_TIP]
        left = landmarks[self.LEFT_EAR_POINT]
        right = landmarks[self.RIGHT_EAR_POINT]
        face_width = distance.euclidean(left, right)
        face_center_x = (left[0] + right[0]) / 2
        return abs(nose[0] - face_center_x) / (face_width + 1e-6)

    def analyze_frame(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        default = {
            "face_detected": False,
            "ear": 0.35, "mar": 0.0, "distraction": 0.0,
            "eye_closed": False, "yawning": False, "distracted": False,
        }

        if not results.multi_face_landmarks:
            return default

        lm = self._landmarks_to_np(results.multi_face_landmarks[0].landmark, w, h)

        left_eye = lm[self.LEFT_EYE]
        right_eye = lm[self.RIGHT_EYE]
        mouth = lm[self.MOUTH]

        ear = (self._ear(left_eye) + self._ear(right_eye)) / 2.0
        mar = self._mar(mouth)
        distraction = self._distraction(lm)

        return {
            "face_detected": True,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "distraction": round(distraction, 3),
            "eye_closed": ear < Config.EAR_THRESHOLD,
            "yawning": mar > Config.MAR_THRESHOLD,
            "distracted": distraction > Config.DISTRACTION_THRESHOLD,
        }