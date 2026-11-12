import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np

project_root = r"D:\Phuong Space\personal documents\Project tieng Nghe An\VASA_Game"
bank_path = os.path.join(project_root, "models", "native_reference_bank.npz")
data = np.load(bank_path)

train_w2v = data["train_w2v"]              # (280, 768)
train_mfcc = data["train_mfcc"]            # (280, 26)
train_prosody = data["train_prosody"]      # (280, 6)
train_f0 = data["train_f0"]                # (280, 200)

val_w2v = data["validation_w2v"]          # (40, 768)
val_mfcc = data["validation_mfcc"]        # (40, 26)
val_prosody = data["validation_prosody"]  # (40, 6)
val_f0 = data["validation_f0"]            # (40, 200)

centroid_w2v = data["centroid_w2v"]        # (768,)
pca_components = data["pca_components"]    # (32, 768)
pca_mean = data["pca_mean"]                # (768,)
pca_var = data["pca_variance"]            # (32,)
mfcc_mean = data["mfcc_mean"]              # (26,)
mfcc_std = data["mfcc_std"]                # (26,)
prosody_mean = data["prosody_mean"]        # (6,)
prosody_std = data["prosody_std"]          # (6,)


# 1. Method A: Cosine Similarity to Native Centroid
def score_cosine_centroid(w2v_emb):
    sim = float(np.dot(w2v_emb, centroid_w2v))
    # Calibrate: Native median ~0.988, min ~0.95
    # Linear/sigmoid mapping: 0.95 -> 70, 0.988 -> 88, 0.995 -> 96
    score = np.clip((sim - 0.92) / (0.995 - 0.92) * 100.0, 0.0, 100.0)
    return score, sim


# 2. Method B: Top-K Nearest Neighbor Native Cosine Similarity
def score_topk_cosine(w2v_emb, k=5):
    sims = np.dot(train_w2v, w2v_emb)
    topk_sims = np.sort(sims)[-k:]
    mean_sim = float(np.mean(topk_sims))
    score = np.clip((mean_sim - 0.93) / (0.995 - 0.93) * 100.0, 0.0, 100.0)
    return score, mean_sim


# 3. Method C: Regularized Mahalanobis Distance in PCA Subspace
def score_mahalanobis(w2v_emb, reg=1e-3):
    diff = w2v_emb - pca_mean
    z = np.dot(pca_components, diff) # (32,)
    d2 = np.sum((z ** 2) / (pca_var + reg))
    d = np.sqrt(d2)
    # Expected chi dimension = sqrt(32) ≈ 5.65
    # Similarity decays exponentially with Mahalanobis distance
    score = 100.0 * np.exp(-0.25 * max(0.0, d - 2.5))
    return float(np.clip(score, 0.0, 100.0)), float(d)


# 4. Method D: Fused Multi-Modal Native Reference Similarity (Wav2Vec2 + F0 + MFCC)
def score_fused(w2v_emb, mfcc_feat, prosody_feat, f0_contour):
    # a. Top-K W2V similarity
    sims = np.dot(train_w2v, w2v_emb)
    topk_sim = float(np.mean(np.sort(sims)[-5:]))
    w2v_score = np.clip((topk_sim - 0.93) / (0.995 - 0.93) * 100.0, 0.0, 100.0)

    # b. Mahalanobis distance in PCA subspace
    diff = w2v_emb - pca_mean
    z = np.dot(pca_components, diff)
    d = float(np.sqrt(np.sum((z ** 2) / (pca_var + 1e-3))))
    maha_score = 100.0 * np.exp(-0.25 * max(0.0, d - 2.5))

    # c. Acoustic MFCC similarity to native distribution
    z_mfcc = (mfcc_feat - mfcc_mean) / mfcc_std
    mfcc_dist = float(np.mean(np.abs(z_mfcc)))
    mfcc_score = 100.0 * np.exp(-0.35 * max(0.0, mfcc_dist - 0.6))

    # d. F0 Modulation Naturalness
    z_prosody = (prosody_feat - prosody_mean) / prosody_std
    prosody_dist = float(np.mean(np.abs(z_prosody)))
    prosody_score = 100.0 * np.exp(-0.40 * max(0.0, prosody_dist - 0.5))

    fused_score = (
        0.45 * w2v_score +
        0.25 * maha_score +
        0.15 * mfcc_score +
        0.15 * prosody_score
    )
    return float(np.clip(round(fused_score, 1), 0.0, 100.0)), {
        "w2v_score": round(w2v_score, 1),
        "maha_score": round(maha_score, 1),
        "mfcc_score": round(mfcc_score, 1),
        "prosody_score": round(prosody_score, 1),
        "topk_sim": round(topk_sim, 4),
        "maha_d": round(d, 3)
    }


print("=== EVALUATING ON VALIDATION SET (40 NATIVE SAMPLES) ===")
scores_a = [score_cosine_centroid(v)[0] for v in val_w2v]
scores_b = [score_topk_cosine(v)[0] for v in val_w2v]
scores_c = [score_mahalanobis(v)[0] for v in val_w2v]
scores_d = [score_fused(val_w2v[i], val_mfcc[i], val_prosody[i], val_f0[i])[0] for i in range(len(val_w2v))]

print(f"Method A (Cosine Centroid)   : Mean={np.mean(scores_a):.1f}, Median={np.median(scores_a):.1f}, Std={np.std(scores_a):.1f}, Min={np.min(scores_a):.1f}, Max={np.max(scores_a):.1f}")
print(f"Method B (Top-K Cosine)      : Mean={np.mean(scores_b):.1f}, Median={np.median(scores_b):.1f}, Std={np.std(scores_b):.1f}, Min={np.min(scores_b):.1f}, Max={np.max(scores_b):.1f}")
print(f"Method C (Reg. Mahalanobis)  : Mean={np.mean(scores_c):.1f}, Median={np.median(scores_c):.1f}, Std={np.std(scores_c):.1f}, Min={np.min(scores_c):.1f}, Max={np.max(scores_c):.1f}")
print(f"Method D (Multi-Feature Fuse): Mean={np.mean(scores_d):.1f}, Median={np.median(scores_d):.1f}, Std={np.std(scores_d):.1f}, Min={np.min(scores_d):.1f}, Max={np.max(scores_d):.1f}")


# 5. Check Sentence-to-Sentence Native Pairwise Matching (When same sentence is spoken)
print("\n=== SENTENCE-SPECIFIC NATIVE RECONSTRUCTION F0 ===")
sample_fnames = ["37_0348.wav", "37_0003.wav", "37_0315.wav", "37_0070.wav"]
for fname in sample_fnames:
    # check where file is located across all splits
    for sp in ["train", "validation", "test"]:
        f_arr = list(data[f"{sp}_filenames"])
        if fname in f_arr:
            idx = f_arr.index(fname)
            w2v = data[f"{sp}_w2v"][idx]
            f0 = data[f"{sp}_f0"][idx]
            txt = data[f"{sp}_texts"][idx]
            print(f"  Found {fname} in {sp} [{txt[:40]}...] -> F0 shape={f0.shape}, W2V shape={w2v.shape}")
            break
