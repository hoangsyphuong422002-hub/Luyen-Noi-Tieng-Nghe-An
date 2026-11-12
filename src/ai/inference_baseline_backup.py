import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import numpy as np

try:
    from src.ai.model import ProsodyAutoEncoder
    from src.ai.preprocess import extract_f0_with_metadata
except ImportError:
    from model import ProsodyAutoEncoder
    from preprocess import extract_f0_with_metadata


class VASAInferenceEngine:
    """
    Upgraded Inference Engine for Vinh Accent Prosody Assessment (VASA).
    - Strict multi-gate Voice Activity & Pitch Validation (rejects silence, noise, clicks, hums).
    - Uses Conv1D ProsodyAutoEncoder with scale-invariant Normalized MSE (NMSE),
      Derivative (Slope) Correlation, and Prosodic Modulation profiling.
    """
    def __init__(self, model_path: str = None, stats_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        
        if model_path is None:
            model_path = os.path.join(project_root, "models", "nghean_ae.pth")
        if stats_path is None:
            stats_path = os.path.join(project_root, "models", "prosody_stats.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                self.stats = json.load(f)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        input_dim = checkpoint.get("input_dim", 200)
        latent_dim = checkpoint.get("latent_dim", 32)

        self.model = ProsodyAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.input_dim = input_dim

    def compute_prosody_score(
        self,
        mse_loss: float,
        original_f0: np.ndarray,
        reconstructed_f0: np.ndarray,
        meta: dict
    ) -> dict:
        """
        Computes a calibrated 0-100 score combining:
        1. Scale-invariant Normalized MSE (NMSE = MSE / Var(x))
        2. Raw MSE against native benchmark (median ~ 1.49)
        3. Contour & Tonal-Slope Correlation
        4. Prosodic Modulation Penalty (penalizes monotone or unnatural pitch range)
        """
        var_f0 = float(np.var(original_f0)) + 1e-6
        std_f0 = float(np.std(original_f0))
        nmse = mse_loss / var_f0

        # 1. NMSE Score (Native median NMSE ~ 0.34, P90 ~ 0.55)
        # If NMSE <= 0.18 -> ~100; NMSE == 0.34 -> ~85; NMSE >= 0.90 -> <25
        eff_nmse = max(0.0, nmse - 0.15)
        nmse_score = 100.0 * np.exp(-1.45 * eff_nmse)

        # 2. Absolute MSE Score (Native median MSE ~ 1.49)
        eff_mse = max(0.0, mse_loss - 0.50)
        mse_score = 100.0 * np.exp(-0.16 * eff_mse)

        # 3. Contour & Slope Correlation
        orig_c = original_f0 - np.mean(original_f0)
        recon_c = reconstructed_f0 - np.mean(reconstructed_f0)
        corr = float(np.dot(orig_c, recon_c) / (np.linalg.norm(orig_c) * np.linalg.norm(recon_c) + 1e-8))
        
        d_orig = np.diff(original_f0)
        d_recon = np.diff(reconstructed_f0)
        slope_corr = float(np.dot(d_orig, d_recon) / (np.linalg.norm(d_orig) * np.linalg.norm(d_recon) + 1e-8))

        corr_combined = 0.6 * max(0.0, corr) + 0.4 * max(0.0, slope_corr)
        corr_score = 100.0 * (corr_combined ** 1.2)

        # 4. Prosodic Modulation Multiplier (Native STD is typically 1.8 - 5.0 semitones)
        # Penalize if voice is too monotonous (std < 1.6) or overly erratic (std > 7.5)
        if std_f0 < 1.6:
            mod_multiplier = np.clip((std_f0 - 0.8) / 0.8, 0.25, 1.0)
        elif std_f0 > 7.5:
            mod_multiplier = np.clip(1.0 - (std_f0 - 7.5) * 0.1, 0.4, 1.0)
        else:
            mod_multiplier = 1.0

        raw_score = (0.45 * nmse_score + 0.30 * mse_score + 0.25 * corr_score) * float(mod_multiplier)
        final_score = float(np.clip(round(raw_score, 1), 0.0, 100.0))

        # Rank & Feedback
        if final_score >= 85:
            rank = "Dân Nghệ Chính Gốc (S-Rank)"
            feedback = "Chuẩn đét luôn! Độ nhấn nhá và luyến trầm đặc trưng xứ Nghệ cực kỳ tự nhiên!"
        elif final_score >= 70:
            rank = "Rất Giống Giọng Nghệ (A-Rank)"
            feedback = "Khá lắm! Đường cong cao độ của bạn bám rất sát quy luật trầm bổng của người bản địa."
        elif final_score >= 50:
            rank = "Đang Tập Nói Tiếng Nghệ (B-Rank)"
            feedback = "Đã có nét giống nhưng một số từ luyến chưa tới. Hãy nhìn đường màu xanh của AI để nhấn trầm các dấu nặng/hỏi hơn nhé!"
        else:
            rank = "Khách Phương Xa (C-Rank)"
            feedback = "Ngữ điệu của bạn còn lệch nhiều so với giọng Nghệ An (hoặc còn hơi đều giọng). Hãy nghe kỹ file mẫu rồi thử lại nhé!"

        return {
            "score": final_score,
            "nmse": round(nmse, 4),
            "rank": rank,
            "feedback": feedback
        }

    def evaluate_audio(self, audio_input) -> dict:
        """
        Full pipeline: Audio -> VAD & F0 Extraction -> Conv1D Auto-Encoder -> Calibrated Score.
        """
        meta = extract_f0_with_metadata(audio_input, target_len=self.input_dim)
        original_f0 = meta["contour"]

        # Immediate rejection if no valid human speech or prosody detected!
        if not meta["is_valid_speech"]:
            zeros = [0.0] * self.input_dim
            return {
                "score": 0.0,
                "mse_loss": None,
                "rank": "Không Phát Hiện Giọng Nói",
                "feedback": meta["reject_reason"],
                "original_f0": zeros,
                "reconstructed_f0": zeros,
                "is_valid_speech": False
            }

        with torch.no_grad():
            tensor_in = torch.tensor(original_f0, dtype=torch.float32).unsqueeze(0).to(self.device)
            tensor_out = self.model(tensor_in)
            reconstructed_f0 = tensor_out.squeeze(0).cpu().numpy()
            mse_loss = float(torch.mean((tensor_out - tensor_in) ** 2).item())

        eval_info = self.compute_prosody_score(mse_loss, original_f0, reconstructed_f0, meta)

        return {
            "score": eval_info["score"],
            "mse_loss": round(mse_loss, 4),
            "nmse": eval_info["nmse"],
            "rank": eval_info["rank"],
            "feedback": eval_info["feedback"],
            "original_f0": [round(float(x), 3) for x in original_f0],
            "reconstructed_f0": [round(float(x), 3) for x in reconstructed_f0],
            "is_valid_speech": True
        }


_engine_instance = None

def get_engine() -> VASAInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = VASAInferenceEngine()
    return _engine_instance

def score_audio_file(audio_input) -> dict:
    engine = get_engine()
    return engine.evaluate_audio(audio_input)
