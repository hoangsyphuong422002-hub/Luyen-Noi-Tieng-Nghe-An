import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure VASA_Game is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import csv
import json
import time
import numpy as np
from sklearn.decomposition import PCA

from src.ai.feature_extractor import SpeechFeatureExtractor, WAV2VEC2_MODEL_NAME


def build_native_reference_bank(project_root: str):
    dataset_dir = os.path.join(project_root, "dataset")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    audios_dir = os.path.join(dataset_dir, "audios")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    # Read dataset records
    records_by_split = {"train": [], "validation": [], "test": []}
    with open(metadata_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train").strip().lower()
            fname = row.get("filename", "").strip()
            if not fname.endswith(".wav"):
                fname += ".wav"
            fpath = os.path.join(audios_dir, fname)
            if os.path.exists(fpath):
                rec = {
                    "filename": fname,
                    "filepath": fpath,
                    "text": row.get("text", "").strip(),
                    "speakerID": row.get("speakerID", "").strip(),
                    "gender": row.get("gender", "").strip(),
                    "split": split
                }
                if split in records_by_split:
                    records_by_split[split].append(rec)
                else:
                    records_by_split["train"].append(rec)

    extractor = SpeechFeatureExtractor.get_instance()
    bank_data = {}

    for split in ["train", "validation", "test"]:
        file_list = records_by_split[split]
        print(f"\n[ReferenceBank] Extracting representations for '{split}' ({len(file_list)} files)...")
        t0 = time.time()
        
        w2v_list = []
        mfcc_list = []
        prosody_list = []
        f0_list = []
        valid_filenames = []
        valid_texts = []
        valid_speakers = []

        for i, rec in enumerate(file_list, 1):
            feats = extractor.extract_features(rec["filepath"])
            if feats["is_valid_speech"]:
                w2v_list.append(feats["w2v_embedding"])
                mfcc_list.append(feats["mfcc_features"])
                prosody_list.append(feats["prosody_features"])
                f0_list.append(feats["f0_contour"])
                valid_filenames.append(rec["filename"])
                valid_texts.append(rec["text"])
                valid_speakers.append(rec["speakerID"])
            else:
                print(f"  Warning: Invalid speech in {rec['filename']} ({feats['reject_reason']})")
                
            if i % 50 == 0 or i == len(file_list):
                elapsed = time.time() - t0
                print(f"  Processed {i}/{len(file_list)} in {elapsed:.1f}s ({elapsed/i:.2f}s/file)")

        bank_data[f"{split}_w2v"] = np.array(w2v_list, dtype=np.float32)
        bank_data[f"{split}_mfcc"] = np.array(mfcc_list, dtype=np.float32)
        bank_data[f"{split}_prosody"] = np.array(prosody_list, dtype=np.float32)
        bank_data[f"{split}_f0"] = np.array(f0_list, dtype=np.float32)
        bank_data[f"{split}_filenames"] = np.array(valid_filenames, dtype=str)
        bank_data[f"{split}_texts"] = np.array(valid_texts, dtype=str)
        bank_data[f"{split}_speakers"] = np.array(valid_speakers, dtype=str)

        # Cache split checkpoint
        split_cache = os.path.join(models_dir, f"split_cache_{split}.npz")
        np.savez_compressed(
            split_cache,
            w2v=bank_data[f"{split}_w2v"],
            mfcc=bank_data[f"{split}_mfcc"],
            prosody=bank_data[f"{split}_prosody"],
            f0=bank_data[f"{split}_f0"],
            filenames=bank_data[f"{split}_filenames"],
            texts=bank_data[f"{split}_texts"],
            speakers=bank_data[f"{split}_speakers"]
        )

    # Compute Native Reference Distribution strictly on Train set (no leakage)
    train_w2v = bank_data["train_w2v"]
    centroid_w2v = np.mean(train_w2v, axis=0)
    centroid_w2v = centroid_w2v / (np.linalg.norm(centroid_w2v) + 1e-8)
    bank_data["centroid_w2v"] = centroid_w2v.astype(np.float32)

    # Fit PCA (32 components) on Train Wav2Vec2 representations
    n_components = min(32, len(train_w2v))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_w2v)
    bank_data["pca_components"] = pca.components_.astype(np.float32)
    bank_data["pca_mean"] = pca.mean_.astype(np.float32)
    bank_data["pca_variance"] = pca.explained_variance_.astype(np.float32)

    # Acoustic feature normalizers from Train
    train_mfcc = bank_data["train_mfcc"]
    train_prosody = bank_data["train_prosody"]
    bank_data["mfcc_mean"] = np.mean(train_mfcc, axis=0).astype(np.float32)
    bank_data["mfcc_std"] = (np.std(train_mfcc, axis=0) + 1e-6).astype(np.float32)
    bank_data["prosody_mean"] = np.mean(train_prosody, axis=0).astype(np.float32)
    bank_data["prosody_std"] = (np.std(train_prosody, axis=0) + 1e-6).astype(np.float32)

    # Compute pairwise native similarity benchmark on Train
    train_sims = np.dot(train_w2v, train_w2v.T)
    # Exclude diagonal
    np.fill_diagonal(train_sims, np.nan)
    native_sim_vals = train_sims[~np.isnan(train_sims)]

    # Compute validation similarities to Train centroid
    val_w2v = bank_data["validation_w2v"]
    val_sims = np.dot(val_w2v, centroid_w2v)

    metadata = {
        "train_samples": len(train_w2v),
        "validation_samples": len(val_w2v),
        "test_samples": len(bank_data["test_w2v"]),
        "w2v_dim": int(train_w2v.shape[1]),
        "pca_dim": int(n_components),
        "pca_explained_var_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "native_pairwise_sim_mean": float(np.mean(native_sim_vals)),
        "native_pairwise_sim_median": float(np.median(native_sim_vals)),
        "native_pairwise_sim_std": float(np.std(native_sim_vals)),
        "native_pairwise_sim_p10": float(np.percentile(native_sim_vals, 10)),
        "native_pairwise_sim_p90": float(np.percentile(native_sim_vals, 90)),
        "val_to_centroid_sim_mean": float(np.mean(val_sims)),
        "val_to_centroid_sim_median": float(np.median(val_sims)),
        "model_name": WAV2VEC2_MODEL_NAME
    }

    # Save to disk
    npz_path = os.path.join(models_dir, "native_reference_bank.npz")
    json_path = os.path.join(models_dir, "native_reference_metadata.json")
    
    np.savez_compressed(npz_path, **bank_data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[ReferenceBank] Saved native reference bank to {npz_path}")
    print(f"[ReferenceBank] Metadata: {json_path}")
    print(f"  Native Train Pairwise Similarity: Mean={metadata['native_pairwise_sim_mean']:.4f}, Median={metadata['native_pairwise_sim_median']:.4f}, P10={metadata['native_pairwise_sim_p10']:.4f}, P90={metadata['native_pairwise_sim_p90']:.4f}")
    print(f"  Validation To Centroid Sim: Mean={metadata['val_to_centroid_sim_mean']:.4f}, Median={metadata['val_to_centroid_sim_median']:.4f}")

    return npz_path, json_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    build_native_reference_bank(project_root)
