# Note: I apologize in advance for the messy code and lack of comments.

# This is for founder personality analysis inference.

import os
import cv2
import torch
import whisper
import torchaudio
import tempfile
from mediapipe.python.solutions import face_mesh, face_detection
class mp:
    solutions = type("solutions", (), {})()
    solutions.face_mesh = face_mesh
    solutions.face_detection = face_detection
import numpy as np
import math
import re
import base64
from PIL import Image as PILImage
from io import BytesIO
from torchvision import transforms as T
from deepface import DeepFace
import google.generativeai as genai
from buildPlusModel6 import MultimodalTransformerModel, VisionEncoder, Wav2Vec2Finetune, TransformerFusion
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import gdown


TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
VC_TRAITS = ["Vision", "Grit", "Coachability", "Charisma", "Execution", "Emotional Stability"]
LABEL_MEAN = np.array([0.56628148, 0.52273139, 0.47614642, 0.54818132, 0.52028646])
LABEL_STD = np.array([0.14697756, 0.1552065, 0.15228453, 0.13637365, 0.15353347])
MAX_FRAMES = 30
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

MODEL_PATH = "finetuned_multimodal_model6.pth"
GOOGLE_DRIVE_ID = "1IcFlwlC9fcN5kpPIeDka12SAMby3JZkm"  # where the model weights are stored
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

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

def extract_faces(video_path, detector, max_faces=16):
    if video_path is None:
        return []
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

def extract_face_sequence(video_path, max_frames=MAX_FRAMES):
    if video_path is None:
        # Return zero tensor if no video
        return torch.zeros(max_frames, 3, 224, 224)
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
    if video_path is None:
        # Return zero tensor if no video
        return torch.zeros(target_len)
    temp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    cmd = f"ffmpeg -y -i '{video_path}' -ar 16000 -ac 1 -f wav {temp_audio} -loglevel error"
    result = os.system(cmd)
    if result != 0 or not os.path.exists(temp_audio):
        # Return zeros on failure
        return torch.zeros(target_len)
    try:
        waveform, _ = torchaudio.load(temp_audio)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
        else:
            waveform = waveform[:target_len]
    except Exception:
        waveform = torch.zeros(target_len)
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
    return waveform

def analyze_emotions(face_images):
    if not face_images:
        return []
    emotions = []
    for img in face_images:
        try:
            res = DeepFace.analyze(np.array(img), actions=["emotion"], enforce_detection=False)
            emotions.append(res[0]['dominant_emotion'])
        except Exception:
            emotions.append("Unknown")
    return emotions

def summarize_audio(video_path, processor, wav2vec_model):
    if video_path is None:
        return ""
    temp_audio = os.path.join(tempfile.gettempdir(), "summary_audio.wav")
    cmd = f"ffmpeg -y -i '{video_path}' -ar 16000 -ac 1 -f wav {temp_audio} -loglevel error"
    result = os.system(cmd)
    if result != 0 or not os.path.exists(temp_audio):
        return ""
    try:
        waveform, sample_rate = torchaudio.load(temp_audio)
        waveform = waveform.mean(dim=0) if waveform.ndim == 2 else waveform.squeeze(0)
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            features = wav2vec_model(**inputs).last_hidden_state
            mean_features = features.mean(dim=1).squeeze()
            variance = mean_features.var().item()
        if variance >= 0.0333:
            summary = "Expressive vocal tone with strong emotional variation."
        elif 0.0283 <= variance < 0.0333:
            summary = "Moderate vocal variation with balanced delivery."
        elif 0.0241 <= variance < 0.0283:
            summary = "Calm and steady tone with limited inflection."
        else:
            summary = "Flat or monotone vocal tone."
        return f"{summary} (Var: {variance:.3f})"
    except Exception:
        return ""
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

def pil_image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def send_to_gemini(company_name, transcript, audio_summary, faces, emotions, predicted_traits):
    parts = []

    # Add face images to Gemini parts
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
You are a VC analyst. Evaluate the startup founder of the company "{company_name}" using their video pitch.

Here is the relevant information:

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

2. Then provide a **detailed qualitative assessment** of the founder’s communication style and emotional signals. You may want to also consider transcript content, but keep in mind this is **NOT** a pitch. Focus primarily on the founder's personality and characteristics that hint towards likelihood of leading a successful startup. Articulate pros and cons of the founder's personality.

3. Finally, provide a **clear recommendation** for next steps from an investor's perspective.

Do NOT repeat the VC traits scores in the qualitative assessment or recommendation sections.
"""

    # Final message part (prompt)
    parts.append({"text": prompt_text})

    # Generate response
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
                try:
                    score = float(score_str.strip())
                except ValueError:
                    score = None
                reasoning = reason_part.strip()
                vc_traits[trait_name] = (score, reasoning)

    cleaned_report = re.sub(r"VC TRAITS WITH REASONING:.*?(?:\n\s*\n|$)", "", full_text, flags=re.DOTALL).strip()

    return vc_traits, cleaned_report

def run_personality_analysis(video_path, transcript, company_name):
    whisper_model, processor, wav2vec_model, face_detector, personality_model = load_models()

    face_imgs = extract_faces(video_path, face_detector)
    face_seq = extract_face_sequence(video_path)
    audio_seq = extract_audio_sequence(video_path)
    emotions = analyze_emotions(face_imgs)
    audio_summary = summarize_audio(video_path, processor, wav2vec_model)

    face_tensor = face_seq.unsqueeze(0).to(DEVICE)
    audio_tensor = audio_seq.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = personality_model(face_tensor, audio_tensor).squeeze().cpu().numpy()

    preds = np.clip(preds, 0, 1)
    z_scores = (preds - LABEL_MEAN) / LABEL_STD
    flags = ["Outlier" if abs(z) >= 2 else "Less Common" if abs(z) >= 1 else "" for z in z_scores]

    vc_traits, qualitative = send_to_gemini(company_name, transcript, audio_summary, face_imgs, emotions, preds)
    vc_scores = {k: v[0] for k, v in vc_traits.items()}

    return {
        "big_five": preds.tolist(),
        "big_five_flags": flags,
        "vc_scores": vc_scores,
        "vc_traits": vc_traits,       
        "qualitative": qualitative.strip(),
        "audio_summary": audio_summary,
        "emotions": emotions,
        "face_images": face_imgs
    }

