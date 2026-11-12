import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import io
import csv
import numpy as np
import soundfile as sf
import parselmouth
from scipy.signal import medfilt, savgol_filter


def extract_f0_with_metadata(
    audio_input,
    target_len: int = 200,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    f_ref: float = 100.0
) -> dict:
    """
    Extracts normalized 200-point F0 contour AND voice activity / prosody metadata.
    
    Includes:
    - Audio RMS energy check & DC offset removal
    - Stricter Praat voicing thresholds to reject background room noise
    - Trimming of leading/trailing unvoiced silence so 200 points cover only active speech
    - Median filtering & Savitzky-Golay smoothing to eliminate octave-jump artifacts
    
    Returns dict with:
        - contour: np.ndarray of shape (target_len,), float32
        - is_valid_speech: bool
        - reject_reason: str or None
        - rms_energy: float
        - voiced_frames: int
        - voiced_ratio: float
        - active_duration: float
        - pitch_std: float
        - pitch_range: float
    """
    # 1. Load audio into numpy array + parselmouth.Sound
    if isinstance(audio_input, (str, os.PathLike)):
        data, sr = sf.read(str(audio_input))
    elif isinstance(audio_input, bytes):
        data, sr = sf.read(io.BytesIO(audio_input))
    elif hasattr(audio_input, "read"):
        data, sr = sf.read(audio_input)
    elif isinstance(audio_input, tuple):
        data, sr = audio_input
    elif isinstance(audio_input, parselmouth.Sound):
        sound = audio_input
        data = sound.values[0]
        sr = int(sound.sampling_frequency)
    else:
        raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float64)

    # Remove DC offset
    if len(data) > 0:
        data = data - np.mean(data)

    # Compute overall & active RMS energy
    rms_energy = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
    peak_amp = float(np.max(np.abs(data))) if len(data) > 0 else 0.0

    zero_contour = np.zeros(target_len, dtype=np.float32)

    # Gate 1: Minimum acoustic energy (reject dead mic / silence)
    if rms_energy < 0.004 or peak_amp < 0.025:
        return {
            "contour": zero_contour,
            "is_valid_speech": False,
            "reject_reason": "Không phát hiện giọng nói (Âm lượng quá nhỏ hoặc bạn chưa nói gì).",
            "rms_energy": rms_energy,
            "voiced_frames": 0,
            "voiced_ratio": 0.0,
            "active_duration": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0
        }

    # Normalize peak amplitude to 0.9 before Praat analysis so mic gain doesn't skew silence threshold,
    # ONLY AFTER verifying rms_energy and peak_amp are above noise floor!
    # Also check SNR / crest structure: in pure white/pink noise amplified by AGC, voicing is low.
    norm_data = data / (peak_amp + 1e-8) * 0.9
    sound = parselmouth.Sound(norm_data, sampling_frequency=sr)

    # Use stricter silence_threshold (0.08) and voicing_threshold (0.50) to reject room noise
    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor,
        max_number_of_candidates=15,
        very_accurate=False,
        silence_threshold=0.08,
        voicing_threshold=0.50,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
        pitch_ceiling=pitch_ceiling
    )
    f0 = pitch.selected_array["frequency"].copy()

    voiced_mask = f0 > 0
    voiced_frames = int(np.sum(voiced_mask))

    # Gate 2: Minimum number of voiced frames (at least 0.30s of actual vocal vibration)
    if voiced_frames < 30:
        return {
            "contour": zero_contour,
            "is_valid_speech": False,
            "reject_reason": f"Chưa phát hiện đủ tiếng nói (Chỉ thu được {voiced_frames * 0.01:.2f}s nguyên âm). Hãy đọc trọn câu nhé!",
            "rms_energy": rms_energy,
            "voiced_frames": voiced_frames,
            "voiced_ratio": 0.0,
            "active_duration": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0
        }

    # Trim leading and trailing unvoiced frames so contour spans the actual spoken utterance
    voiced_indices = np.where(voiced_mask)[0]
    start_idx = voiced_indices[0]
    end_idx = voiced_indices[-1]
    span_len = end_idx - start_idx + 1
    active_duration = float(span_len * time_step)
    voiced_ratio = float(voiced_frames / max(1, span_len))

    # Gate 3: Minimum active speech span and voiced density
    if active_duration < 0.50 or voiced_ratio < 0.18:
        return {
            "contour": zero_contour,
            "is_valid_speech": False,
            "reject_reason": "Tín hiệu giọng nói quá rời rạc hoặc chỉ có tạp âm nền. Vui lòng nói rõ ràng vào micro!",
            "rms_energy": rms_energy,
            "voiced_frames": voiced_frames,
            "voiced_ratio": voiced_ratio,
            "active_duration": active_duration,
            "pitch_std": 0.0,
            "pitch_range": 0.0
        }

    # Crop to active utterance span
    f0_cropped = f0[start_idx:end_idx + 1]
    voiced_cropped = f0_cropped > 0

    # Convert voiced F0 to semitones
    voiced_f0_vals = f0_cropped[voiced_cropped]
    voiced_semitones = 12.0 * np.log2(voiced_f0_vals / f_ref)

    # Apply 5-point median filter on voiced semitones to eliminate single-frame octave jumps
    if len(voiced_semitones) >= 5:
        voiced_semitones = medfilt(voiced_semitones, kernel_size=5)

    # Mean-normalize (remove individual speaker pitch height)
    mean_st = float(np.mean(voiced_semitones))
    voiced_semitones = voiced_semitones - mean_st

    # Clip extreme hardware/tracker glitches to [-14, +14] semitones
    voiced_semitones = np.clip(voiced_semitones, -14.0, 14.0)

    pitch_std = float(np.std(voiced_semitones))
    pitch_range = float(np.percentile(voiced_semitones, 95) - np.percentile(voiced_semitones, 5))

    # Gate 4: Minimum pitch variation (reject electrical hum 50/100Hz or flat buzz)
    if pitch_std < 0.85 or pitch_range < 2.5:
        return {
            "contour": zero_contour,
            "is_valid_speech": False,
            "reject_reason": "Giọng thu âm hoàn toàn phẳng (đơn âm hoặc nhiễu quạt/điện). Hãy nói có ngữ điệu trầm bổng nhé!",
            "rms_energy": rms_energy,
            "voiced_frames": voiced_frames,
            "voiced_ratio": voiced_ratio,
            "active_duration": active_duration,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range
        }

    # Interpolate across unvoiced consonants within the active utterance
    all_cropped_idx = np.arange(len(f0_cropped))
    voiced_cropped_idx = np.where(voiced_cropped)[0]
    continuous_semitones = np.interp(all_cropped_idx, voiced_cropped_idx, voiced_semitones)

    # Resample to fixed target_len (200 points)
    orig_time = np.linspace(0.0, 1.0, len(continuous_semitones))
    target_time = np.linspace(0.0, 1.0, target_len)
    fixed_contour = np.interp(target_time, orig_time, continuous_semitones)

    # Gentle Savitzky-Golay smoothing on the 200-point curve to remove linear interpolation corners
    if target_len >= 11:
        fixed_contour = savgol_filter(fixed_contour, window_length=11, polyorder=3)

    # Re-center to exact zero mean
    fixed_contour = fixed_contour - np.mean(fixed_contour)

    return {
        "contour": fixed_contour.astype(np.float32),
        "is_valid_speech": True,
        "reject_reason": None,
        "rms_energy": rms_energy,
        "voiced_frames": voiced_frames,
        "voiced_ratio": voiced_ratio,
        "active_duration": active_duration,
        "pitch_std": float(np.std(fixed_contour)),
        "pitch_range": float(np.percentile(fixed_contour, 95) - np.percentile(fixed_contour, 5))
    }


def extract_f0_contour(audio_input, target_len: int = 200) -> np.ndarray:
    """Backward-compatible wrapper returning just the 200-point numpy array."""
    res = extract_f0_with_metadata(audio_input, target_len=target_len)
    return res["contour"]


def process_dataset(dataset_dir: str, output_dir: str, target_len: int = 200):
    """
    Reads metadata.csv from dataset_dir, extracts cleaned 200-pt F0 contours for all splits,
    and saves X_train.npy, X_val.npy, and X_test.npy.
    """
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    audios_dir = os.path.join(dataset_dir, "audios")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    records = {"train": [], "validation": [], "test": []}
    
    with open(metadata_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train").strip().lower()
            fname = row.get("filename", "").strip()
            if not fname.endswith(".wav"):
                fname += ".wav"
            fpath = os.path.join(audios_dir, fname)
            if os.path.exists(fpath):
                if split in records:
                    records[split].append((fname, fpath))
                else:
                    records["train"].append((fname, fpath))

    for split_name, file_list in records.items():
        print(f"\n[Preprocess] Processing split '{split_name}' ({len(file_list)} files)...")
        features = []
        valid_count = 0
        for i, (fname, fpath) in enumerate(file_list, 1):
            try:
                meta = extract_f0_with_metadata(fpath, target_len=target_len)
                if meta["is_valid_speech"]:
                    features.append(meta["contour"])
                    valid_count += 1
                else:
                    print(f"  [Skipped invalid/noisy file] {fname}: {meta['reject_reason']}")
                if i % 50 == 0 or i == len(file_list):
                    print(f"  Processed {i}/{len(file_list)} (valid: {valid_count})")
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
                
        matrix = np.array(features, dtype=np.float32)
        out_name = f"X_{split_name if split_name != 'validation' else 'val'}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, matrix)
        print(f"-> Saved {out_name} with shape {matrix.shape} to {out_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    process_dataset(dataset_dir, project_root, target_len=200)
