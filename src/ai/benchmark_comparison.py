import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import csv
import io
import wave
import numpy as np
import soundfile as sf

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
AUDIOS_DIR = os.path.join(DATASET_DIR, "audios")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ai.inference_baseline_backup import VASAInferenceEngine as BaselineEngine
from src.ai.inference import VASAInferenceEngine as UpgradedEngine


def generate_edge_case_audio(case_type: str, duration_sec: float = 2.0, sr: int = 16000) -> str:
    """Creates a temporary in-memory WAV or file for edge-case testing."""
    n_samples = int(duration_sec * sr)
    if case_type == "silence":
        audio = np.zeros(n_samples, dtype=np.float32)
    elif case_type == "white_noise":
        audio = np.random.uniform(-0.05, 0.05, n_samples).astype(np.float32)
    elif case_type == "sine_monotone_220hz":
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    elif case_type == "click_impulse":
        audio = np.zeros(n_samples, dtype=np.float32)
        audio[sr // 2] = 0.9  # Single impulse in the middle
    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    out_path = os.path.join(PROJECT_ROOT, "models", f"test_edge_{case_type}.wav")
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path


def load_dataset_splits():
    splits = {"train": [], "val": [], "test": []}
    with open(METADATA_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train").strip().lower()
            if split == "validation":
                split = "val"
            fname = row.get("filename", "").strip()
            fpath = os.path.join(AUDIOS_DIR, fname)
            if os.path.exists(fpath) and split in splits:
                splits[split].append((fname, fpath))
    return splits


def run_benchmark():
    print("=" * 70)
    print("VASA BENCHMARK: Baseline (AutoEncoder) vs Upgraded (Wav2Vec2 + Fusion)")
    print("=" * 70)

    print("\n[1/4] Initializing Baseline Engine (Conv1D AutoEncoder)...")
    base_engine = BaselineEngine()
    print("[1/4] Baseline Engine ready.")

    print("\n[2/4] Initializing Upgraded Engine (Pretrained Wav2Vec2 + Acoustic Fusion)...")
    up_engine = UpgradedEngine.get_instance()
    print("[2/4] Upgraded Engine ready.")

    splits = load_dataset_splits()
    print(f"\n[3/4] Loaded dataset: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test.")

    # Results dictionary
    results = {
        "baseline": {"val": [], "test": [], "edge": [], "latencies": []},
        "upgraded_general": {"val": [], "test": [], "edge": [], "latencies": []},
        "upgraded_sentence_matched": {"val": [], "test": []}
    }

    # Evaluate Validation Set (40 samples)
    print("\nEvaluating Validation Set (40 samples)...")
    for fname, fpath in splits["val"]:
        # Baseline
        t0 = time.perf_counter()
        res_base = base_engine.evaluate_audio(fpath)
        lat_base = (time.perf_counter() - t0) * 1000
        results["baseline"]["val"].append(res_base["score"])
        results["baseline"]["latencies"].append(lat_base)

        # Upgraded General Mode
        t0 = time.perf_counter()
        res_up_gen = up_engine.evaluate_audio(fpath, reference_audio=None)
        lat_up = (time.perf_counter() - t0) * 1000
        results["upgraded_general"]["val"].append(res_up_gen["score"])
        results["upgraded_general"]["latencies"].append(lat_up)

        # Upgraded Sentence-Matched Mode (self-reference benchmark)
        res_up_match = up_engine.evaluate_audio(fpath, reference_audio=fname)
        results["upgraded_sentence_matched"]["val"].append(res_up_match["score"])

    # Evaluate Test Set (43 samples)
    print("Evaluating Test Set (43 samples)...")
    for fname, fpath in splits["test"]:
        res_base = base_engine.evaluate_audio(fpath)
        results["baseline"]["test"].append(res_base["score"])

        res_up_gen = up_engine.evaluate_audio(fpath, reference_audio=None)
        results["upgraded_general"]["test"].append(res_up_gen["score"])

        res_up_match = up_engine.evaluate_audio(fpath, reference_audio=fname)
        results["upgraded_sentence_matched"]["test"].append(res_up_match["score"])

    # Edge cases
    print("\n[4/4] Evaluating Edge Cases (Silence, White Noise, Monotone 220Hz, Click)...")
    edge_types = ["silence", "white_noise", "sine_monotone_220hz", "click_impulse"]
    edge_report = []

    for c in edge_types:
        edge_file = generate_edge_case_audio(c)
        r_base = base_engine.evaluate_audio(edge_file)
        r_up = up_engine.evaluate_audio(edge_file, reference_audio=None)

        edge_report.append({
            "case": c,
            "baseline_score": r_base["score"],
            "baseline_valid": r_base.get("is_valid_speech", False),
            "baseline_rank": r_base.get("rank"),
            "upgraded_score": r_up["score"],
            "upgraded_valid": r_up.get("is_valid_speech", False),
            "upgraded_rank": r_up.get("rank"),
            "upgraded_feedback": r_up.get("feedback")
        })

    # Summary Statistics Calculation
    def stats(arr):
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 2),
            "median": round(float(np.median(a)), 2),
            "std": round(float(np.std(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2)
        }

    summary = {
        "val_set": {
            "n": len(splits["val"]),
            "baseline": stats(results["baseline"]["val"]),
            "upgraded_general": stats(results["upgraded_general"]["val"]),
            "upgraded_sentence_matched": stats(results["upgraded_sentence_matched"]["val"])
        },
        "test_set": {
            "n": len(splits["test"]),
            "baseline": stats(results["baseline"]["test"]),
            "upgraded_general": stats(results["upgraded_general"]["test"]),
            "upgraded_sentence_matched": stats(results["upgraded_sentence_matched"]["test"])
        },
        "latency_ms": {
            "baseline_mean": round(float(np.mean(results["baseline"]["latencies"])), 2),
            "baseline_p50": round(float(np.median(results["baseline"]["latencies"])), 2),
            "baseline_p95": round(float(np.percentile(results["baseline"]["latencies"], 95)), 2),
            "upgraded_mean": round(float(np.mean(results["upgraded_general"]["latencies"])), 2),
            "upgraded_p50": round(float(np.median(results["upgraded_general"]["latencies"])), 2),
            "upgraded_p95": round(float(np.percentile(results["upgraded_general"]["latencies"], 95)), 2)
        },
        "edge_cases": edge_report
    }

    out_json = os.path.join(PROJECT_ROOT, "models", "benchmark_comparison_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print("\n--- VALIDATION SET (40 Native Speakers) ---")
    print(f"Baseline AE:               Mean={summary['val_set']['baseline']['mean']}, Med={summary['val_set']['baseline']['median']}, Std={summary['val_set']['baseline']['std']} (Min={summary['val_set']['baseline']['min']}, Max={summary['val_set']['baseline']['max']})")
    print(f"Upgraded (General Mode):   Mean={summary['val_set']['upgraded_general']['mean']}, Med={summary['val_set']['upgraded_general']['median']}, Std={summary['val_set']['upgraded_general']['std']} (Min={summary['val_set']['upgraded_general']['min']}, Max={summary['val_set']['upgraded_general']['max']})")
    print(f"Upgraded (Sentence-Match): Mean={summary['val_set']['upgraded_sentence_matched']['mean']}, Med={summary['val_set']['upgraded_sentence_matched']['median']}, Std={summary['val_set']['upgraded_sentence_matched']['std']}")

    print("\n--- TEST SET (43 Native Speakers) ---")
    print(f"Baseline AE:               Mean={summary['test_set']['baseline']['mean']}, Med={summary['test_set']['baseline']['median']}, Std={summary['test_set']['baseline']['std']} (Min={summary['test_set']['baseline']['min']}, Max={summary['test_set']['baseline']['max']})")
    print(f"Upgraded (General Mode):   Mean={summary['test_set']['upgraded_general']['mean']}, Med={summary['test_set']['upgraded_general']['median']}, Std={summary['test_set']['upgraded_general']['std']} (Min={summary['test_set']['upgraded_general']['min']}, Max={summary['test_set']['upgraded_general']['max']})")
    print(f"Upgraded (Sentence-Match): Mean={summary['test_set']['upgraded_sentence_matched']['mean']}, Med={summary['test_set']['upgraded_sentence_matched']['median']}, Std={summary['test_set']['upgraded_sentence_matched']['std']}")

    print("\n--- INFERENCE LATENCY (ms per sample on CPU) ---")
    print(f"Baseline AE:     Mean={summary['latency_ms']['baseline_mean']}ms, P50={summary['latency_ms']['baseline_p50']}ms, P95={summary['latency_ms']['baseline_p95']}ms")
    print(f"Upgraded W2V:    Mean={summary['latency_ms']['upgraded_mean']}ms, P50={summary['latency_ms']['upgraded_p50']}ms, P95={summary['latency_ms']['upgraded_p95']}ms")

    print("\n--- EDGE CASE REJECTION ---")
    for e in edge_report:
        print(f"[{e['case']}] Baseline: score={e['baseline_score']}, valid={e['baseline_valid']} | Upgraded: score={e['upgraded_score']}, valid={e['upgraded_valid']} ({e['upgraded_rank']})")

    print(f"\nFull results saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
