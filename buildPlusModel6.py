# I apologize in advance for the messy code and lack of comments

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
import pickle
import cv2
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T
# import mediapipe as mp
from mediapipe.python.solutions import face_mesh, face_detection
class mp:
    solutions = type("solutions", (), {})()
    solutions.face_mesh = face_mesh
    solutions.face_detection = face_detection
import math
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MAX_SAMPLES = 100
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 42
MAX_FRAMES = 30
MODEL_SAVE_PATH = "finetuned_multimodal_model.pth"
ROOT_DIR = "/Users/matthewkim/Documents/Build+/train-1"
ANNOT_PATH = os.path.join(ROOT_DIR, "annotation_training.pkl")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED)

def load_labels(annot_path):
    with open(annot_path, 'rb') as f:
        ann = pickle.load(f, encoding='latin1')
    data = {}
    for vid in ann['openness'].keys():
        data[vid] = [ann[k][vid] for k in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]
    return data

def find_videos_and_labels(root, label_dict):
    video_paths, labels = [], []
    for subfolder in os.listdir(root):
        full_sub = os.path.join(root, subfolder)
        if not os.path.isdir(full_sub) or "annotation" in subfolder:
            continue
        for fname in os.listdir(full_sub):
            if fname.endswith(".mp4") and fname in label_dict:
                video_paths.append(os.path.join(full_sub, fname))
                labels.append(label_dict[fname])
    return video_paths, labels

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# ---- [MODIFIED] Face Alignment ----
def extract_face_crops(video_path, max_frames=30):
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
            eye_l = landmarks.landmark[33]
            eye_r = landmarks.landmark[263]
            dx = (eye_r.x - eye_l.x) * w
            dy = (eye_r.y - eye_l.y) * h
            angle = -math.degrees(math.atan2(dy, dx))

            cx, cy = int(nose.x * w), int(nose.y * h)
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            aligned = cv2.warpAffine(frame, M, (w, h))

            bbox = landmarks.landmark[234:454]
            xs = [int(p.x * w) for p in bbox]
            ys = [int(p.y * h) for p in bbox]
            x1, y1, x2, y2 = max(min(xs), 0), max(min(ys), 0), min(max(xs), w), min(max(ys), h)
            face = aligned[y1:y2, x1:x2]
            if face.size > 0:
                face = transform(face)
                frames.append(face)
        count += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(torch.zeros(3, 224, 224))
    return torch.stack(frames[:max_frames])

class Wav2Vec2Finetune(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", output_dim=512):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.norm = nn.LayerNorm(output_dim)
        self.proj = nn.Linear(768, output_dim).to(device)

        for name, param in self.model.named_parameters():
            if "encoder.layers.10" in name or "encoder.layers.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, waveform):
        waveform_cpu = waveform.detach().cpu().numpy()
        list_waveforms = [w for w in waveform_cpu]
        inputs = self.processor(list_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}
        x = self.model(**inputs).last_hidden_state
        pooled = x.mean(dim=1, keepdim=True)
        return self.norm(self.proj(pooled))

def extract_audio_waveform(video_path):
    audio_path = "temp.wav"
    os.system(f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 1 -f wav -y {audio_path} > /dev/null 2>&1")
    waveform, _ = torchaudio.load(audio_path)
    return waveform.squeeze(0)

def build_dataset(video_paths, labels, max_samples=MAX_SAMPLES):
    face_seqs, audio_seqs, targets = [], [], []
    for i in tqdm(range(min(max_samples, len(video_paths))), desc="Extracting features"):
        try:
            face_seq = extract_face_crops(video_paths[i]).unsqueeze(0)
            audio_seq = extract_audio_waveform(video_paths[i])
            face_seqs.append(face_seq)
            audio_seqs.append(audio_seq)
            targets.append(torch.tensor(labels[i], dtype=torch.float32))
        except Exception as e:
            print(f"Failed on {video_paths[i]}: {e}")
    return face_seqs, audio_seqs, targets

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in base.parameters():
            param.requires_grad = False
        for name, param in base.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
        base.fc = nn.Linear(base.fc.in_features, output_dim)
        self.cnn = base

    def forward(self, face_seq):
        B, T, C, H, W = face_seq.shape
        x = face_seq.view(B * T, C, H, W)
        x = self.cnn(x)
        return x.view(B, T, -1)
    
# ---- [NEW] Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---- [MODIFIED] Transformer Fusion ----
class TransformerFusion(nn.Module):
    def __init__(self, dim=512, heads=8, seq_len=60):
        super().__init__()
        self.pe = PositionalEncoding(dim, max_len=seq_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True),
            num_layers=2)

    def forward(self, vision_seq, audio_seq):
        x = torch.cat([vision_seq, audio_seq], dim=1)  # B x T x D
        x = self.pe(x)
        x = self.transformer(x)
        return x.mean(dim=1)

class MultimodalTransformerModel(nn.Module):
    def __init__(self, vision_encoder, audio_encoder, fusion_module, dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.fusion = fusion_module
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, face_seq, audio_waveform):
        vision_feat = self.vision_encoder(face_seq)
        audio_feat = self.audio_encoder(audio_waveform)
        fused = self.fusion(vision_feat, audio_feat)
        return self.head(fused)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # weight for MSE vs Pearson
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # MSE component
        mse_loss = self.mse(pred, target)

        # Pearson correlation component
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=0)
        denominator = pred_centered.norm(dim=0) * target_centered.norm(dim=0) + 1e-8
        pearson_corr = numerator / denominator
        pearson_loss = -pearson_corr.mean()  # negative for minimization

        # Combined weighted loss
        return self.alpha * mse_loss + (1 - self.alpha) * pearson_loss

# ---- [MODIFIED] Training Loop w/ Early Stopping ----
def train_model(Xf, Xa, Y, patience=5):
    bins = np.digitize([y.mean().item() for y in Y], bins=[0.3, 0.5, 0.7])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(Xf, bins))

    vision_encoder = VisionEncoder()
    audio_encoder = Wav2Vec2Finetune()
    fusion_module = TransformerFusion()

    model = MultimodalTransformerModel(vision_encoder, audio_encoder, fusion_module)
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_idx)//BATCH_SIZE, epochs=EPOCHS)
    loss_fn = CombinedLoss(alpha=0.5)

    best_val = float('inf')
    patience_counter = 0

    def batch_loader(indices):
        for i in range(0, len(indices), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            face = torch.cat([Xf[j].to(device) for j in idx])
            audio = torch.stack([Xa[j] for j in idx]).to(device)
            y = torch.stack([Y[j] for j in idx]).to(device)
            yield face, audio, y

    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0
        for face, audio, y in batch_loader(train_idx):
            optimizer.zero_grad()
            preds = model(face, audio)
            losses = [loss_fn(preds[:, i], y[:, i]) for i in range(5)]
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train += loss.item() * len(y)

        model.eval()
        loss_val = 0
        with torch.no_grad():
            for face, audio, y in batch_loader(val_idx):
                preds = model(face, audio)
                loss_val += loss_fn(preds, y).item() * len(y)

        train_loss = loss_train / len(train_idx)
        val_loss = loss_val / len(val_idx)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print("✅ Best model saved.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for face, audio, y in batch_loader(val_idx):
            preds = model(face, audio).clamp(0, 1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\n🔍 Example predictions (first 5):")
    for i in range(min(5, len(all_preds))):
        print(f"  Predicted: {np.round(all_preds[i], 2)} | True: {np.round(all_labels[i], 2)}")

    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    print("\n📈 Metrics per trait:")
    for i, name in enumerate(trait_names):
        mse = mean_squared_error(all_labels[:, i], all_preds[:, i])
        corr = np.corrcoef(all_labels[:, i], all_preds[:, i])[0, 1]
        print(f"  {name}: MSE = {mse:.4f}, Pearson r = {corr:.4f}")


    return model

if __name__ == "__main__":
    print("📦 Loading labels...")
    labels = load_labels(ANNOT_PATH)
    print("🎥 Finding videos...")
    video_paths, Y = find_videos_and_labels(ROOT_DIR, labels)
    print(f"Found {len(video_paths)} labeled videos.")
    print("🔧 Extracting dataset...")
    Xf, Xa, Y = build_dataset(video_paths, Y)
    print("🧠 Training model...")
    train_model(Xf, Xa, Y)