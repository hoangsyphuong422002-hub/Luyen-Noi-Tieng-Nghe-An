import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import io
import soundfile as sf
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from src.ai.preprocess import extract_f0_with_metadata

WAV2VEC2_MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

class SpeechFeatureExtractor:
    """
    Unified feature extractor combining:
    1. Pretrained Vietnamese Wav2Vec2 representations (phonetic & phonological manifold)
    2. Praat Parselmouth F0 pitch contour & prosodic dynamics
    3. Acoustic timbre features (MFCC, spectral centroid, spectral rolloff)
    4. Audio quality and voice activity validation (VAD)
    """
    _instance = None

    def __init__(self, model_name: str = WAV2VEC2_MODEL_NAME, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[FeatureExtractor] Loading {model_name} on {self.device}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.wav2vec2.eval()
        print("[FeatureExtractor] Pretrained Vietnamese Wav2Vec2 ready.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_audio(self, audio_input, target_sr: int = 16000):
        """Loads audio input into float32 mono array at target_sr (16kHz)."""
        if isinstance(audio_input, (str, os.PathLike)):
            data, sr = sf.read(str(audio_input))
        elif isinstance(audio_input, bytes):
            data, sr = sf.read(io.BytesIO(audio_input))
        elif hasattr(audio_input, "read"):
            data, sr = sf.read(audio_input)
        elif isinstance(audio_input, tuple):
            data, sr = audio_input
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return data, sr

    def extract_features(self, audio_input, max_duration_sec: float = 12.0) -> dict:
        """
        Extracts speech representations, acoustic metrics, and VAD metadata.
        Returns dict with:
            - is_valid_speech: bool
            - reject_reason: str or None
            - w2v_embedding: np.ndarray (768,) L2-normalized mean-pooled embedding
            - w2v_std: np.ndarray (768,) std-pooled embedding
            - mfcc_features: np.ndarray (26,) mean and std of 13 MFCCs
            - prosody_features: np.ndarray (6,) pitch dynamics
            - f0_contour: np.ndarray (200,) normalized F0 contour
            - confidence: float (0.0 to 1.0)
            - duration: float
        """
        # 1. First run multi-gate VAD & F0 extraction
        vad_res = extract_f0_with_metadata(audio_input, target_len=200)
        if not vad_res["is_valid_speech"]:
            return {
                "is_valid_speech": False,
                "reject_reason": vad_res["reject_reason"],
                "w2v_embedding": np.zeros(768, dtype=np.float32),
                "w2v_std": np.zeros(768, dtype=np.float32),
                "mfcc_features": np.zeros(26, dtype=np.float32),
                "prosody_features": np.zeros(6, dtype=np.float32),
                "f0_contour": np.zeros(200, dtype=np.float32),
                "confidence": 0.0,
                "duration": 0.0
            }

        # 2. Load normalized audio for Wav2Vec2 & MFCC
        data_16k, sr = self.load_audio(audio_input, target_sr=16000)
        total_duration = len(data_16k) / sr

        # Truncate to max_duration_sec if excessively long
        max_samples = int(max_duration_sec * sr)
        audio_chunk = data_16k[:max_samples]

        # 3. Wav2Vec2 Feature Extraction (Frozen Encoder)
        inputs = self.feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden_states = self.wav2vec2(**inputs).last_hidden_state[0]  # (T, 768)
            w2v_mean = torch.mean(hidden_states, dim=0).cpu().numpy()
            w2v_std = torch.std(hidden_states, dim=0).cpu().numpy()

        # L2-normalize mean embedding for cosine similarity
        norm_val = np.linalg.norm(w2v_mean) + 1e-8
        w2v_emb_norm = (w2v_mean / norm_val).astype(np.float32)

        # 4. Acoustic Features (MFCC 13 + dynamics)
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=16000, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_feats = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)

        # 5. Prosody Summary Vector
        prosody_feats = np.array([
            vad_res["pitch_std"],
            vad_res["pitch_range"],
            vad_res["voiced_ratio"],
            vad_res["active_duration"],
            vad_res["rms_energy"] * 100.0,
            float(np.std(np.diff(vad_res["contour"]))) # slope variance
        ], dtype=np.float32)

        # 6. Quality & Confidence Score
        # Audio between 1.0s and 8.0s with good voiced ratio has high confidence
        dur_score = float(np.clip(vad_res["active_duration"] / 2.0, 0.4, 1.0))
        voiced_score = float(np.clip(vad_res["voiced_ratio"] / 0.35, 0.5, 1.0))
        energy_score = float(np.clip(vad_res["rms_energy"] / 0.02, 0.5, 1.0))
        confidence = float(np.clip(dur_score * 0.4 + voiced_score * 0.35 + energy_score * 0.25, 0.2, 1.0))

        return {
            "is_valid_speech": True,
            "reject_reason": None,
            "w2v_embedding": w2v_emb_norm,
            "w2v_std": w2v_std.astype(np.float32),
            "mfcc_features": mfcc_feats,
            "prosody_features": prosody_feats,
            "f0_contour": vad_res["contour"],
            "confidence": round(confidence, 3),
            "duration": round(total_duration, 2)
        }
