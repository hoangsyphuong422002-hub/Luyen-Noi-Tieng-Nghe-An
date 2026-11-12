import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import numpy as np
from src.ai.feature_extractor import SpeechFeatureExtractor


class VASAInferenceEngine:
    """
    Upgraded AI Scoring Engine for Vinh Accent Prosody Assessment (VASA).
    
    Architecture:
    1. Pretrained Vietnamese Wav2Vec2 (nguyenvulebinh/wav2vec2-base-vietnamese-250h)
       for phonetic, phonological, and regional pronunciation representation.
    2. Praat Parselmouth 200-point F0 normalized pitch contour for tonal prosody.
    3. Multi-gate Voice Activity Detection (VAD) & acoustic feature fusion (MFCC + dynamics).
    4. Native Nghệ An Reference Bank similarity scoring:
       - Sentence-aware comparison when practicing a specific challenge sentence.
       - Manifold similarity (Top-K Cosine + Regularized Mahalanobis) when evaluating general speech.
    """
    _instance = None

    def __init__(self, project_root: str = None):
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

        self.project_root = project_root
        self.models_dir = os.path.join(project_root, "models")
        self.bank_path = os.path.join(self.models_dir, "native_reference_bank.npz")
        self.meta_path = os.path.join(self.models_dir, "native_reference_metadata.json")

        if not os.path.exists(self.bank_path):
            raise FileNotFoundError(f"Native reference bank not found at {self.bank_path}. Run build_reference_bank.py first.")

        # 1. Initialize Feature Extractor (Singleton)
        self.extractor = SpeechFeatureExtractor.get_instance()

        # 2. Load Native Reference Bank
        print(f"[InferenceEngine] Loading native reference bank from {self.bank_path}...")
        bank_npz = np.load(self.bank_path)
        self.train_w2v = bank_npz["train_w2v"]              # (280, 768)
        self.train_mfcc = bank_npz["train_mfcc"]            # (280, 26)
        self.train_prosody = bank_npz["train_prosody"]      # (280, 6)
        self.train_f0 = bank_npz["train_f0"]                # (280, 200)

        self.centroid_w2v = bank_npz["centroid_w2v"]        # (768,)
        self.pca_components = bank_npz["pca_components"]    # (32, 768)
        self.pca_mean = bank_npz["pca_mean"]                # (768,)
        self.pca_var = bank_npz["pca_variance"]            # (32,)

        self.mfcc_mean = bank_npz["mfcc_mean"]              # (26,)
        self.mfcc_std = bank_npz["mfcc_std"]                # (26,)
        self.prosody_mean = bank_npz["prosody_mean"]        # (6,)
        self.prosody_std = bank_npz["prosody_std"]          # (6,)

        # Build in-memory lookup index for all 363 native recordings
        self.reference_index = {}
        for split in ["train", "validation", "test"]:
            fnames = bank_npz[f"{split}_filenames"]
            w2v_arr = bank_npz[f"{split}_w2v"]
            mfcc_arr = bank_npz[f"{split}_mfcc"]
            prosody_arr = bank_npz[f"{split}_prosody"]
            f0_arr = bank_npz[f"{split}_f0"]
            texts = bank_npz[f"{split}_texts"]
            for i, fn in enumerate(fnames):
                self.reference_index[fn] = {
                    "w2v": w2v_arr[i],
                    "mfcc": mfcc_arr[i],
                    "prosody": prosody_arr[i],
                    "f0": f0_arr[i],
                    "text": str(texts[i]),
                    "split": split
                }

        self.metadata = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"[InferenceEngine] Ready. Indexed {len(self.reference_index)} native speech references.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _clean_ref_name(self, ref_str: str) -> str:
        if not ref_str:
            return None
        # Handles paths like '/dataset_audios/37_0348.wav' or '37_0348.wav' or '37_0348'
        base = os.path.basename(str(ref_str)).strip()
        if not base.endswith(".wav") and "." not in base:
            base += ".wav"
        return base

    def evaluate_audio(self, audio_input, reference_audio: str = None) -> dict:
        """
        Processes audio input, extracts representations, and scores native similarity.
        
        Args:
            audio_input: File path, bytes, or file-like audio buffer.
            reference_audio: Optional filename of the challenge sentence native recording.
            
        Returns dict formatted compatibly with the existing VASA frontend:
            - score: float (0.0 - 100.0)
            - rank: str (S-Rank, A-Rank, B-Rank, C-Rank)
            - feedback: str
            - similarity_index: float (0.0 - 1.0)
            - confidence: float (0.0 - 1.0)
            - reliability_status: str
            - original_f0: list of 200 floats
            - reconstructed_f0: list of 200 floats (target native F0)
            - mse_loss: float (relative distance metric for frontend display)
            - is_valid_speech: bool
        """
        # 1. Feature Extraction & Multi-Gate VAD Check
        feats = self.extractor.extract_features(audio_input)

        if not feats["is_valid_speech"]:
            zeros = [0.0] * 200
            return {
                "score": 0.0,
                "mse_loss": None,
                "nmse": None,
                "similarity_index": 0.0,
                "confidence": 0.0,
                "reliability_status": "Không hợp lệ",
                "rank": "Không Phát Hiện Giọng Nói",
                "feedback": feats["reject_reason"],
                "original_f0": zeros,
                "reconstructed_f0": zeros,
                "is_valid_speech": False,
                "model_type": "Pretrained Wav2Vec2-Vietnamese + Acoustic Fusion"
            }

        user_w2v = feats["w2v_embedding"]       # (768,)
        user_mfcc = feats["mfcc_features"]      # (26,)
        user_prosody = feats["prosody_features"]# (6,)
        user_f0 = feats["f0_contour"]           # (200,)
        confidence = feats["confidence"]

        clean_ref = self._clean_ref_name(reference_audio)
        ref_entry = self.reference_index.get(clean_ref) if clean_ref else None

        # 2. Case A: Sentence-Matched Scoring (Exact Target Reference Available)
        if ref_entry is not None:
            ref_w2v = ref_entry["w2v"]
            ref_mfcc = ref_entry["mfcc"]
            ref_f0 = ref_entry["f0"]

            # a. Wav2Vec2 Phonetic & Pronunciation Similarity (Cosine)
            w2v_sim = float(np.dot(user_w2v, ref_w2v))

            # b. Prosodic F0 Contour Correlation & Slope Alignment
            u_c = user_f0 - np.mean(user_f0)
            r_c = ref_f0 - np.mean(ref_f0)
            f0_corr = float(np.dot(u_c, r_c) / (np.linalg.norm(u_c) * np.linalg.norm(r_c) + 1e-8))

            du = np.diff(user_f0)
            dr = np.diff(ref_f0)
            slope_corr = float(np.dot(du, dr) / (np.linalg.norm(du) * np.linalg.norm(dr) + 1e-8))
            prosody_sim = 0.60 * max(0.0, f0_corr) + 0.40 * max(0.0, slope_corr)

            # c. Acoustic MFCC distance
            z_diff = (user_mfcc - ref_mfcc) / self.mfcc_std
            mfcc_dist = float(np.mean(np.abs(z_diff)))
            mfcc_sim = float(np.exp(-0.35 * max(0.0, mfcc_dist - 0.4)))

            # Fused Sentence-Level Similarity Index (0.0 to 1.0)
            sim_index = 0.50 * w2v_sim + 0.35 * prosody_sim + 0.15 * mfcc_sim

            # Calibrate to 0-100 Relative Similarity Index
            # Native identical speech yields sim_index ~ 0.95 - 0.99 (Score 90-100)
            # Acceptable approximation yields sim_index ~ 0.75 - 0.85 (Score 65-80)
            calibrated = (sim_index - 0.65) / (0.97 - 0.65) * 100.0
            final_score = float(np.clip(round(calibrated, 1), 0.0, 100.0))

            # Target contour displayed in the frontend chart is the exact native speaker contour!
            reconstructed_f0 = [round(float(x), 3) for x in ref_f0]
            comparison_mode = f"Đối chiếu trực tiếp câu mẫu ({clean_ref})"

        # 3. Case B: General Manifold Scoring (Unconstrained Native Reference Bank)
        else:
            # a. Top-5 Nearest Native Neighbors Cosine Similarity
            all_sims = np.dot(self.train_w2v, user_w2v)
            topk_idx = np.argsort(all_sims)[-5:]
            topk_sim = float(np.mean(all_sims[topk_idx]))
            w2v_score = np.clip((topk_sim - 0.93) / (0.995 - 0.93) * 100.0, 0.0, 100.0)

            # b. Regularized Mahalanobis Distance in PCA Subspace
            diff = user_w2v - self.pca_mean
            z = np.dot(self.pca_components, diff)
            maha_d = float(np.sqrt(np.sum((z ** 2) / (self.pca_var + 1e-3))))
            maha_score = 100.0 * np.exp(-0.25 * max(0.0, maha_d - 2.5))

            # c. Acoustic MFCC similarity to native distribution
            z_mfcc = (user_mfcc - self.mfcc_mean) / self.mfcc_std
            mfcc_score = 100.0 * np.exp(-0.35 * max(0.0, float(np.mean(np.abs(z_mfcc))) - 0.6))

            # d. Prosody Modulation Naturalness
            z_prosody = (user_prosody - self.prosody_mean) / self.prosody_std
            prosody_score = 100.0 * np.exp(-0.40 * max(0.0, float(np.mean(np.abs(z_prosody))) - 0.5))

            fused_score = (
                0.45 * w2v_score +
                0.25 * maha_score +
                0.15 * mfcc_score +
                0.15 * prosody_score
            )
            final_score = float(np.clip(round(fused_score, 1), 0.0, 100.0))
            sim_index = topk_sim

            # Choose nearest native speaker's F0 contour for visual comparison
            best_native_idx = topk_idx[-1]
            reconstructed_f0 = [round(float(x), 3) for x in self.train_f0[best_native_idx]]
            comparison_mode = "Đối chiếu không gian ngữ âm bản địa (280 mẫu Nghệ An)"

        # 4. Generate Ranking & Pedagogical Feedback
        if final_score >= 85:
            rank = "Dân Nghệ Chính Gốc (S-Rank)"
            feedback = "Rất tuyệt vời! Ngữ điệu trầm bổng và âm sắc phát âm mang đặc trưng rất rõ nét của người Nghệ An."
        elif final_score >= 70:
            rank = "Rất Giống Giọng Nghệ (A-Rank)"
            feedback = "Khá lắm! Bạn đã nắm được nhịp điệu và độ nhấn nhá tương đối tự nhiên của người bản địa."
        elif final_score >= 50:
            rank = "Đang Tập Nói Tiếng Nghệ (B-Rank)"
            feedback = "Đã có nét giống nhưng một số từ cao độ còn hơi ngang. Hãy nhìn đường màu xanh để hạ trầm các thanh nặng/hỏi hơn nhé!"
        else:
            rank = "Khách Phương Xa (C-Rank)"
            feedback = "Ngữ điệu của bạn còn khác nhiều so với giọng Nghệ An. Hãy nghe kỹ file mẫu và nhấn nhá theo gợi ý nhé!"

        # Reliability status
        if confidence >= 0.85:
            reliability_status = "Độ tin cậy cao (Bản thu rõ, âm lượng chuẩn)"
        elif confidence >= 0.60:
            reliability_status = "Độ tin cậy khá (Bản thu đủ tiêu chuẩn)"
        else:
            reliability_status = "Độ tin cậy thấp (Bản thu ngắn hoặc có tiếng ồn)"

        return {
            "score": final_score,
            "similarity_index": round(float(sim_index), 4),
            "confidence": confidence,
            "reliability_status": reliability_status,
            "comparison_mode": comparison_mode,
            "rank": rank,
            "feedback": feedback,
            "mse_loss": round(float(1.0 - sim_index), 4),
            "nmse": round(float(sim_index), 4),
            "original_f0": [round(float(x), 3) for x in user_f0],
            "reconstructed_f0": reconstructed_f0,
            "is_valid_speech": True,
            "model_type": "Pretrained Wav2Vec2-Vietnamese + Acoustic Fusion"
        }


# Singleton accessor
def get_engine() -> VASAInferenceEngine:
    return VASAInferenceEngine.get_instance()

def score_audio_file(audio_input, reference_audio: str = None) -> dict:
    engine = get_engine()
    return engine.evaluate_audio(audio_input, reference_audio=reference_audio)


if __name__ == "__main__":
    eng = get_engine()
    # Test 1: Self native test sample with reference
    res1 = eng.evaluate_audio("VASA_Game/dataset/audios/37_0348.wav", reference_audio="37_0348.wav")
    print("Test 1 (Self 37_0348 with ref):", res1["score"], res1["rank"], res1["similarity_index"])

    # Test 2: Native test sample without reference (general mode)
    res2 = eng.evaluate_audio("VASA_Game/dataset/audios/37_0348.wav", reference_audio=None)
    print("Test 2 (General mode 37_0348):", res2["score"], res2["rank"], res2["similarity_index"])

    # Test 3: Native 37_0003 with ref
    res3 = eng.evaluate_audio("VASA_Game/dataset/audios/37_0003.wav", reference_audio="37_0003.wav")
    print("Test 3 (Self 37_0003 with ref):", res3["score"], res3["rank"], res3["similarity_index"])
