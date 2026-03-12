# config.py

class Config:
    # Assets
    ALARM_PATH = "alarm.wav"
    
    # Detection Parameters (Optimized for real-world webcam)
    DROWSY_SCORE_THRESHOLD = 15  
    SCORE_INCREMENT = 3
    SCORE_DECAY = 1
    SCORE_MIN = 0

    # MediaPipe Thresholds (Peak Accuracy)
    EAR_THRESHOLD = 0.22 
    MAR_THRESHOLD = 0.45
    DISTRACTION_THRESHOLD = 0.20

    # Logging
    LOG_FILE = "drowsiness_log.txt"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"