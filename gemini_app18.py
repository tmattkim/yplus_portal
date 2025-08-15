import os
import cv2
import torch
import whisper
import torchaudio
import tempfile
import mediapipe as mp
import numpy as np
import subprocess
import math
import re
import plotly.graph_objects as go
from fpdf import FPDF
from PIL import Image as PILImage
from deepface import DeepFace
import streamlit as st
import google.generativeai as genai
import unicodedata
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import base64
from io import BytesIO
from torchvision import transforms as T
from buildPlusModel6 import MultimodalTransformerModel, VisionEncoder, Wav2Vec2Finetune, TransformerFusion
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import gdown


# ========== CONSTANTS ==========
MODEL_PATH = "finetuned_multimodal_model6.pth"
GOOGLE_DRIVE_ID = "1IcFlwlC9fcN5kpPIeDka12SAMby3JZkm"
LABEL_MEAN = np.array([0.56628148, 0.52273139, 0.47614642, 0.54818132, 0.52028646])
LABEL_STD = np.array([0.14697756, 0.1552065, 0.15228453, 0.13637365, 0.15353347])
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
VC_TRAITS = ["Vision", "Grit", "Coachability", "Charisma", "Execution", "Emotional Stability"]
MAX_FRAMES = 30
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

def download_model(model_path=MODEL_PATH, drive_id=GOOGLE_DRIVE_ID):
    """
    Downloads the model from Google Drive if not already cached locally.
    """
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading to {model_path}...")
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, model_path, quiet=False)
        print("Download complete!")
    else:
        print(f"Model already exists at {model_path}. Using cached version.")

@st.cache_resource
def load_models():
    download_model()
    whisper_model = whisper.load_model("base")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    vision_encoder = VisionEncoder()
    audio_encoder = Wav2Vec2Finetune()
    fusion_module = TransformerFusion()
    personality_model = MultimodalTransformerModel(vision_encoder, audio_encoder, fusion_module)
    personality_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    personality_model.eval().to(DEVICE)

    return whisper_model, processor, wav2vec_model, face_detector, personality_model

def extract_audio_from_video(video_path):
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav.name]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav.name

def transcribe(video_path, whisper_model):
    return whisper_model.transcribe(video_path)["text"]

# def summarize_audio(video_path, processor, wav2vec_model):
#     wav_path = extract_audio_from_video(video_path)
#     waveform, sample_rate = torchaudio.load(wav_path)
#     if waveform.ndim == 2 and waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0)
#     else:
#         waveform = waveform.squeeze(0)
#     inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
#     with torch.no_grad():
#         features = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
#     return ("The speaker uses a dynamic vocal tone with varied inflection and pacing." if features.var().item() > 0.08 else
#             "The speaker has a calm and consistent vocal tone with low variability.")

def summarize_audio(video_path, processor, wav2vec_model):
    wav_path = extract_audio_from_video(video_path)
    waveform, sample_rate = torchaudio.load(wav_path)

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        features = wav2vec_model(**inputs).last_hidden_state
        mean_features = features.mean(dim=1).squeeze()
        variance = mean_features.var().item()

    # More nuanced thresholds
    if variance >= 0.0333:
        summary = "The speaker's vocal tone is highly expressive, with noticeable emotional range and dynamic shifts in pitch and pacing."
    elif 0.0283 <= variance < 0.0333:
        summary = "The speaker exhibits moderate vocal variation, suggesting a balanced delivery with occasional shifts in tone and emphasis."
    elif 0.0241 <= variance < 0.0283:
        summary = "The speaker's vocal tone is calm and steady, with limited variation in inflection or energy."
    else:
        summary = "The speaker demonstrates a flat or monotone vocal tone, with minimal emotional variation or inflection."

    # Optionally add numerical value
    summary += f" (Feature variance: {variance:.3f})"
    return summary

def extract_faces(video_path, detector, max_faces=16):
    cap = cv2.VideoCapture(video_path)
    faces, count = [], 0
    while cap.isOpened() and count < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            for det in results.detections:
                ih, iw, _ = frame.shape
                box = det.location_data.relative_bounding_box
                x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
                face = frame[y:y+h, x:x+w]
                if face.size:
                    faces.append(PILImage.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
                    count += 1
                if count >= max_faces:
                    break
    cap.release()
    return faces

def analyze_emotions(face_images):
    emotions = []
    for img in face_images:
        try:
            res = DeepFace.analyze(np.array(img), actions=["emotion"], enforce_detection=False)
            emotions.append(res[0]['dominant_emotion'])
        except Exception:
            emotions.append("Unknown")
    return emotions

def extract_face_sequence(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            nose = landmarks.landmark[1]
            eye_l, eye_r = landmarks.landmark[33], landmarks.landmark[263]
            dx = (eye_r.x - eye_l.x) * w
            dy = (eye_r.y - eye_l.y) * h
            angle = -math.degrees(math.atan2(dy, dx))
            cx, cy = int(nose.x * w), int(nose.y * h)
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            aligned = cv2.warpAffine(frame, M, (w, h))
            bbox = landmarks.landmark[234:454]
            xs = [int(p.x * w) for p in bbox]
            ys = [int(p.y * h) for p in bbox]
            x1, y1 = max(min(xs), 0), max(min(ys), 0)
            x2, y2 = min(max(xs), w), min(max(ys), h)
            face = aligned[y1:y2, x1:x2]
            if face.size > 0:
                face = transform(face)
                frames.append(face)
        count += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(torch.zeros(3, 224, 224))
    return torch.stack(frames[:max_frames])

def extract_audio_sequence(video_path, target_len=250000):
    temp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    os.system(f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 1 -f wav -y {temp_audio} > /dev/null 2>&1")
    waveform, _ = torchaudio.load(temp_audio)
    waveform = waveform.squeeze(0)
    return waveform[:target_len] if waveform.shape[0] >= target_len else torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))

def pil_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_vc_traits(gemini_text):
    scores = {}
    for trait in VC_TRAITS:
        match = re.search(rf"{trait}\s*:\s*([01]\.\d+)", gemini_text)
        if match:
            scores[trait] = float(match.group(1))
    return scores

def send_to_gemini(transcript, audio_summary, faces, emotions, predicted_traits):
    parts = []

    # Add images as inline_data parts
    for img in faces:
        img_b64 = pil_image_to_base64(img)
        parts.append({
            "inline_data": {
                "data": img_b64,
                "mime_type": "image/png"
            }
        })

    big_five_str = "\n".join([f"{trait}: {score:.2f}" for trait, score in zip(TRAITS, predicted_traits)])
    emotion_tags = ", ".join(emotions)

    prompt_text = f"""
You are a VC analyst. Evaluate this startup founder, particularly their personality, from a first impressions video using the information below:

Transcript:
\"\"\"
{transcript}
\"\"\"

Vocal Summary:
{audio_summary}

Detected Emotions:
{emotion_tags}

Big Five Personality Traits (Predicted from AI model):
{big_five_str}

Instructions:

1. Provide **VC Trait Ratings** (scale 0.00 to 1.00) WITH A REASONING EXPLANATION for each trait **in this exact format**:

VC TRAITS WITH REASONING:
Vision: 0.00 — Reasoning: <explanation here>
Grit: 0.00 — Reasoning: <explanation here>
Coachability: 0.00 — Reasoning: <explanation here>
Charisma: 0.00 — Reasoning: <explanation here>
Execution: 0.00 — Reasoning: <explanation here>
Emotional Stability: 0.00 — Reasoning: <explanation here>

2. Then provide a **detailed qualitative assessment** of the founder’s communication style and emotional signals. You may want to also consider transcript content, but keep in mind this is **NOT** a pitch. Focus primarily on the founder's personality and characteristics that hint towards likelihood of leading a successful startup. Articulate pros and cons of the founder's personality. Focus **ONLY** on the founder's personality in evaluating since startup ideas can pivot and shift. The product, market, or current startup idea is not as important.

3. Finally, provide a **clear recommendation** for next steps from an investor's perspective.

Do NOT repeat the VC traits scores in the qualitative assessment or recommendation sections.
"""

    parts.append({"text": prompt_text})
    content = {"parts": parts}

    response = gemini_model.generate_content(content)
    full_text = response.text

    # Parse VC traits with reasoning block
    vc_traits = {}
    vc_block = re.search(r"VC TRAITS WITH REASONING:(.*?)(?:\n\s*\n|$)", full_text, re.DOTALL)
    if vc_block:
        lines = vc_block.group(1).strip().splitlines()
        for line in lines:
            if "— Reasoning:" in line:
                trait_part, reason_part = line.split("— Reasoning:", 1)
                trait_name, score_str = trait_part.split(":")
                trait_name = trait_name.strip()
                score = float(score_str.strip())
                reasoning = reason_part.strip()
                vc_traits[trait_name] = (score, reasoning)

    # Remove VC traits block from final report
    cleaned_report = re.sub(r"VC TRAITS WITH REASONING:.*?(?:\n\s*\n|$)", "", full_text, flags=re.DOTALL).strip()

    return vc_traits, cleaned_report

# Streamlit UI
# st.set_page_config(page_title="Founder Personality Analyzer", layout="wide")
# st.title("🧠 Founder Personality Analyzer")
def founder_personality_analyzer():
    # st.markdown("Or upload a startup founder video. We'll analyze their personality using multimodal AI and provide an investor-style evaluation.")
    # st.markdown("<small>Or upload a startup founder video. We'll analyze their personality using multimodal AI and provide an investor-style evaluation.</small>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("🎥 Or upload a startup founder recording (MP4). Please ensure only one person is present.", type=["mp4"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        st.video(video_path)
        whisper_model, processor, wav2vec_model, face_detector, personality_model = load_models()

        with st.spinner("Transcribing speech..."):
            transcript = transcribe(video_path, whisper_model)
            st.subheader("📝 Transcript")
            st.markdown(f"<div style='background:#f1f3f4;color:#111;padding:10px;border-radius:10px'>{transcript}</div>", unsafe_allow_html=True)

        with st.spinner("Analyzing vocal tone..."):
            audio_summary = summarize_audio(video_path, processor, wav2vec_model)
            st.markdown(f"🎤 **Vocal Summary**: {audio_summary}")

        with st.spinner("Extracting face crops..."):
            face_images = extract_faces(video_path, face_detector)
            st.subheader("📷 Sample Face Frames")
            st.image(face_images, width=150)

        with st.spinner("Detecting emotions..."):
            emotions = analyze_emotions(face_images)
            st.markdown(f"🧠 **FER+ Emotions**: {', '.join(emotions)}")

        with st.spinner("Predicting Big Five traits..."):
            face_seq = extract_face_sequence(video_path)
            audio_seq = extract_audio_sequence(video_path)
            face_tensor = face_seq.unsqueeze(0).to(DEVICE)
            audio_tensor = audio_seq.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                preds = personality_model(face_tensor, audio_tensor).squeeze().cpu().numpy()
            preds = np.clip(preds, 0, 1)
            z_scores = (preds - LABEL_MEAN) / LABEL_STD

            st.success("✅ Big Five prediction complete!")
            st.subheader("📊 Big Five Trait Scores")
            for i, (trait, value, z) in enumerate(zip(TRAITS, preds, z_scores)):
                st.markdown(f"**{trait}** — Score: `{value:.2f}`")
                st.progress(float(value))
                delta = f"{z:+.2f} SD"
                if abs(z) >= 2:
                    st.warning(f"🚨 Outlier ({delta})")
                elif 2 > abs(z) >= 1:
                    st.warning(f"❗ Less Common ({delta})")
                else:
                    st.caption(f"Deviation from mean: {delta}")
        
        with st.expander("📉 Visualize Big Five Traits Radar Chart"):
            def create_radar_chart(traits, values, title="Big Five Traits", color='rgba(31,119,180,0.7)'):
                fig = go.Figure()

                # Repeat first value to close the radar
                values += values[:1]
                traits += traits[:1]

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=traits,
                    fill='toself',
                    name='Score',
                    line=dict(color=color)
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=False,
                    title=title
                )
                return fig

            bigfive_fig = create_radar_chart(TRAITS.copy(), preds.tolist(), title="Big Five Traits")
            st.plotly_chart(bigfive_fig, use_container_width=True)


        with st.spinner("Generating full report..."):
            vc_traits, result = send_to_gemini(transcript, audio_summary, face_images, emotions, preds)

        st.subheader("🎯 VC Relevant Trait Scores with Reasoning")
        for trait, (score, reasoning) in vc_traits.items():
            st.markdown(f"**{trait}** — Score: `{score:.2f}`")
            st.progress(score)
            st.markdown(f"*Reasoning:* {reasoning}")

        with st.expander("🚀 VC Relevant Traits Radar Chart"):
            vc_values = [score for trait, (score, _) in vc_traits.items()]
            vc_fig = create_radar_chart(VC_TRAITS.copy(), vc_values, title="VC Relevant Traits", color="rgba(255,127,14,0.6)")
            st.plotly_chart(vc_fig, use_container_width=True)

        def format_gemini_report(text):
            # Remove the first sentence or line if it matches the unwanted intro
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("Okay,"):
                lines = lines[1:]  # skip the first line
            # Join back the remaining lines
            text = "\n".join(lines).strip()

            replacements = {
                "**Qualitative Assessment:": "### Qualitative Assessment",
                "**Recommendation:": "### Recommendation",
                "**VC Trait Ratings:": "### VC Trait Ratings",
                # add any other headers you want formatted here
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            # Remove any leftover '**' not part of headers
            text = text.replace("**", "")
            return text

        st.subheader("📈 Qualitative Report")
        formatted_result = format_gemini_report(result)
        st.markdown(formatted_result)

        def clean_text(text):
            return unicodedata.normalize('NFKD', text).encode('latin1', 'ignore').decode('latin1')

        def generate_pdf_report(transcript, audio_summary, emotions, big_five_traits, vc_traits):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("Founder Personality & Communication Report", styles["Title"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("<b>Audio Summary:</b>", styles["Heading2"]))
            story.append(Paragraph(audio_summary, styles["BodyText"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("<b>Emotional Analysis:</b>", styles["Heading2"]))
            story.append(Paragraph(str(emotions), styles["BodyText"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("<b>Big Five Traits:</b>", styles["Heading2"]))
            story.append(Paragraph(", ".join([f"{k}: {v:.2f}" for k, v in big_five_traits.items()]), styles["BodyText"]))
            story.append(Spacer(1, 12))

            # Radar chart for Big Five traits
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(big_five_traits.values()),
                theta=list(big_five_traits.keys()),
                fill='toself',
                name='Big Five Traits'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                width=400,
                height=400,
            )

            radar_buf = BytesIO()
            fig.write_image(radar_buf, format='png')
            radar_buf.seek(0)
            story.append(RLImage(radar_buf, width=300, height=300))
            story.append(Spacer(1, 12))

            # VC Relevant Traits with reasoning
            story.append(Paragraph("<b>VC Relevant Traits with Reasoning:</b>", styles["Heading2"]))
            for trait, (score, reasoning) in vc_traits.items():
                story.append(Paragraph(f"{trait}: {score:.2f}", styles["BodyText"]))
                story.append(Paragraph(f"Reasoning: {reasoning}", styles["BodyText"]))
                story.append(Spacer(1, 6))

            story.append(Spacer(1, 12))

            # Radar chart for VC Relevant Traits
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=[score for score, _ in vc_traits.values()],
                theta=list(vc_traits.keys()),
                fill='toself',
                name='VC Traits'
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                width=400,
                height=400,
            )

            radar_buf2 = BytesIO()
            fig2.write_image(radar_buf2, format='png')
            radar_buf2.seek(0)
            story.append(RLImage(radar_buf2, width=300, height=300))
            story.append(Spacer(1, 12))

            story.append(Paragraph("<b>Transcript:</b>", styles["Heading2"]))
            story.append(Paragraph(transcript, styles["BodyText"]))

            doc.build(story)
            buffer.seek(0)
            return buffer

        traits = {trait: float(score) for trait, score in zip(TRAITS, preds)}

        with st.expander("📄 Download PDF Report"):
            pdf_buffer = generate_pdf_report(transcript, audio_summary, emotions, traits, vc_traits)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_buffer,
                file_name="report.pdf",
                mime="application/pdf"
            )
