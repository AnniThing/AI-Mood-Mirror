import gradio as gr
from fer import FER
import cv2
import numpy as np

detector = FER(mtcnn=True)

# Define colors for different emotions
emotion_colors = {
    "happy": "#22c55e",     # green
    "angry": "#ef4444",     # red
    "sad": "#3b82f6",       # blue
    "surprise": "#eab308",  # yellow
    "fear": "#8b5cf6",      # purple
    "disgust": "#f97316",   # orange
    "neutral": "#64748b",   # gray
}

def analyze_emotion(image):
    if not isinstance(image, np.ndarray):
        return "<div style='color:red; font-weight:bold;'>❌ Error: Invalid image</div>"

    # Detect emotions
    result = detector.detect_emotions(image)

    if not result:
        return "<div style='color:gray; font-weight:bold;'>No face detected. Try another image!</div>"

    # Pick strongest emotion
    emotions = result[0]["emotions"]
    emotion, score = max(emotions.items(), key=lambda x: x[1])

    # Scale confidence to 10
    score_out_of_10 = int(round(score * 10))

    # Pick color for emotion
    color = emotion_colors.get(emotion, "#334155")

    # Styled card output
    return f"""
    <div style='
        background-color: white; 
        border-radius: 16px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;'>
        <h2 style='color:{color}; font-size: 28px; margin-bottom: 10px;'>🎭 Emotion: {emotion.capitalize()}</h2>
        <p style='font-size: 20px; margin: 0;'>📊 Confidence: <b style='color:{color};'>{score_out_of_10}/10</b></p>
    </div>
    """

# --- UI Styling ---
custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    button_primary_background_fill="linear-gradient(90deg, #a855f7, #6366f1)",
    button_primary_background_fill_hover="linear-gradient(90deg, #9333ea, #4f46e5)",
    button_primary_text_color="white",
)

with gr.Blocks(theme=custom_theme, css=".gradio-container {background: #f8fafc}") as demo:
    with gr.Column(elem_id="main-col", scale=1, min_width=600):
        gr.Markdown(
            """
            # 🎭 Emotion Recognition App  
            Upload a photo and let AI detect your **emotions** in real time.  
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload your image", sources=["upload", "webcam"])
                submit_btn = gr.Button("✨ Analyze Emotion", variant="primary")
            with gr.Column():
                output_html = gr.HTML(label="Result")

        submit_btn.click(fn=analyze_emotion, inputs=image_input, outputs=output_html)

demo.launch()