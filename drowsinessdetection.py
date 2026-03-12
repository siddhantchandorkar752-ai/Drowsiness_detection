import cv2
import logging
import numpy as np
from pygame import mixer
from config import Config
from predictor import DrowsinessPredictor

logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def init_alarm():
    mixer.init()
    sound = mixer.Sound(Config.ALARM_PATH)
    logger.info("Alarm initialized.")
    return sound

def play_alarm(sound):
    try:
        if not mixer.get_busy():
            sound.play(-1) # -1 ka matlab hai loop mein bajega jab tak stop na karein
    except mixer.error as e:
        logger.error(f"Alarm failed: {e}")

def stop_alarm(sound):
    if mixer.get_busy():
        sound.stop() # Aankh khulte hi shant karne ka brahmastra

def draw_metric_bar(frame, label, value, y, color, threshold=None):
    bx, bw, bh = 10, 200, 18
    fill = int(min(value, 1.0) * bw)
    cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, y), (bx + fill, y + bh), color, -1)
    cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (200, 200, 200), 1)
    if threshold:
        tx = bx + int(threshold * bw)
        cv2.line(frame, (tx, y), (tx, y + bh), (255, 255, 255), 2)
    cv2.putText(
        frame, f"{label}: {value:.2f}",
        (bx + bw + 10, y + 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (255, 255, 255), 1, cv2.LINE_AA
    )

def draw_hud(frame, metrics, score):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "DROWSINESS MONITOR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 1, cv2.LINE_AA)

    ear_val = max(0.0, 1.0 - (metrics["ear"] / 0.35))
    drowsy_val = min(score / Config.DROWSY_SCORE_THRESHOLD, 1.0)

    draw_metric_bar(frame, "EYE CLOSURE", ear_val, 45, (0, 100, 255), threshold=0.7)
    draw_metric_bar(frame, "DROWSY LVL ", drowsy_val, 75, (0, 50, 255), threshold=1.0)

    status_text = "ALERT & FOCUSED"
    status_color = (0, 160, 0)
    
    if not metrics["face_detected"]:
        status_text, status_color = "NO FACE DETECTED", (80, 80, 80)
    elif metrics["eye_closed"]:
        status_text, status_color = "EYES CLOSED", (0, 0, 200)

    cv2.rectangle(frame, (10, 105), (180, 135), status_color, -1)
    cv2.putText(frame, status_text, (16, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

def update_score(score, metrics):
    if not metrics["face_detected"]:
        return score
    if metrics["eye_closed"]:
        return score + Config.SCORE_INCREMENT
    return max(Config.SCORE_MIN, score - Config.SCORE_DECAY)

def main():
    predictor = DrowsinessPredictor()
    sound = init_alarm()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.critical("Webcam not accessible.")
        return

    score = 0
    logger.info("Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1) # Mirror effect ke liye taaki natural lage
        metrics = predictor.analyze_frame(frame)
        score = update_score(score, metrics)
        
        draw_hud(frame, metrics, score)

        # MAIN LOGIC: Alarm play and instant STOP
        if score >= Config.DROWSY_SCORE_THRESHOLD:
            play_alarm(sound)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
        else:
            stop_alarm(sound) # Aankh khulte hi score girega aur yahan alarm band ho jayega!

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()