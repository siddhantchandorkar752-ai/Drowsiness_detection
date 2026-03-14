import gradio as gr
import numpy as np
import cv2
from predictor import DrowsinessPredictor

predictor = DrowsinessPredictor()

def analyze_frame(image):
    if image is None:
        return {"SYSTEM_ERROR": "NO_INPUT_FEED"}
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    metrics = predictor.analyze_frame(bgr_image)
    if not metrics["face_detected"]:
        return {"SYSTEM_STATUS": "FACE_NOT_FOUND"}
    return {
        "DIAGNOSTIC_DATA": {
            "OPERATOR_STATE": "CRITICAL" if metrics["eye_closed"] else "STABLE",
            "EYE_ASPECT_RATIO": f"{metrics['ear']:.3f}",
            "MOUTH_ASPECT_RATIO": f"{metrics['mar']:.3f}",
            "FOCUS_INDEX": f"{1.0 - metrics['distraction']:.2%}"
        },
        "ACTION_REQUIRED": "WAKE_UP_ALERT" if metrics["eye_closed"] else "NONE"
    }

with gr.Blocks(title="ADMS Neural Monitor") as demo:
    gr.Markdown("# ADMS v2.0 — Neural Monitor\n*High-Precision Biometric Fatigue Analytics Engine*")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(sources=["webcam"], type="numpy", label="NEURAL_FEED_INPUT")
            submit_btn = gr.Button("INITIALIZE SCAN", variant="primary")
        with gr.Column(scale=1):
            output_json = gr.JSON(label="BIOMETRIC_DIAGNOSTICS")
    gr.Markdown("---\nCORE ENGINE: MEDIAPIPE MESH // LATENCY: <15MS")
    submit_btn.click(fn=analyze_frame, inputs=input_img, outputs=output_json)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)