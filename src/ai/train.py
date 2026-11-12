import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model import ProsodyAutoEncoder


def prosody_loss_fn(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined MSE loss + first-order derivative (slope) loss to preserve tonal inflections.
    """
    mse = torch.mean((recon - target) ** 2)
    diff_recon = recon[:, 1:] - recon[:, :-1]
    diff_target = target[:, 1:] - target[:, :-1]
    slope_loss = torch.mean((diff_recon - diff_target) ** 2)
    return mse + 2.0 * slope_loss


def train_autoencoder(
    project_root: str,
    epochs: int = 180,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 32
):
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "nghean_ae.pth")
    stats_path = os.path.join(models_dir, "prosody_stats.json")

    train_path = os.path.join(project_root, "X_train.npy")
    val_path = os.path.join(project_root, "X_val.npy")
    test_path = os.path.join(project_root, "X_test.npy")

    X_train = np.load(train_path)
    X_val = np.load(val_path) if os.path.exists(val_path) else None
    X_test = np.load(test_path) if os.path.exists(test_path) else None

    print(f"Loaded X_train: {X_train.shape}")
    if X_val is not None:
        print(f"Loaded X_val:   {X_val.shape}")
    if X_test is not None:
        print(f"Loaded X_test:  {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = ProsodyAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val_loss = float("inf")

    print("\n--- Starting Training (Conv1D Prosody Auto-Encoder) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_mse = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = prosody_loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            train_mse += mse_criterion(recon, x).item() * len(x)
        
        train_mse /= len(X_train)

        val_mse = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon = model(x)
                    val_mse += mse_criterion(recon, x).item() * len(x)
            val_mse /= len(X_val)
            scheduler.step(val_mse)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "val_loss": best_val_loss,
                    "train_loss": train_mse
                }, best_model_path)
                improved = "*"
            else:
                improved = ""
        else:
            val_mse = train_mse
            improved = ""

        if epoch % 20 == 0 or epoch == 1 or improved == "*":
            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | LR: {curr_lr:.6f} {improved}")

    print("\n--- Training Completed ---")
    print(f"Best Validation MSE: {best_val_loss:.4f}")

    # Load best model & compute detailed benchmark statistics on native speakers
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_matrix = np.vstack([X_val, X_test]) if (X_val is not None and X_test is not None) else X_train
    with torch.no_grad():
        eval_t = torch.tensor(eval_matrix, dtype=torch.float32).to(device)
        recon_t = model(eval_t).cpu().numpy()

    mse_list = np.mean((recon_t - eval_matrix) ** 2, axis=1)
    var_list = np.var(eval_matrix, axis=1) + 1e-6
    nmse_list = mse_list / var_list  # Scale-invariant Normalized MSE!
    std_list = np.std(X_train, axis=1)

    corrs = []
    for i in range(len(eval_matrix)):
        a = eval_matrix[i] - np.mean(eval_matrix[i])
        b = recon_t[i] - np.mean(recon_t[i])
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        corrs.append(c)

    stats = {
        "best_val_loss": float(best_val_loss),
        "test_loss_mean": float(np.mean(mse_list)),
        "test_loss_median": float(np.median(mse_list)),
        "test_loss_p90": float(np.percentile(mse_list, 90)),
        "nmse_mean": float(np.mean(nmse_list)),
        "nmse_median": float(np.median(nmse_list)),
        "nmse_p90": float(np.percentile(nmse_list, 90)),
        "corr_mean": float(np.mean(corrs)),
        "corr_median": float(np.median(corrs)),
        "native_std_mean": float(np.mean(std_list)),
        "native_std_p10": float(np.percentile(std_list, 10)),
        "native_std_p90": float(np.percentile(std_list, 90)),
        "input_dim": input_dim,
        "latent_dim": latent_dim
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Benchmark Native Stats -> MSE Median: {stats['test_loss_median']:.3f}, NMSE Median: {stats['nmse_median']:.3f}, Corr Median: {stats['corr_median']:.3f}")
    return best_model_path, stats


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    train_autoencoder(project_root, epochs=180, batch_size=16, lr=1e-3, latent_dim=32)
