import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from DL.data.generate_synthetic_data import PSDSplitDataset, PSDIntervalDataset, PSDMaskDataset  
from DL.models.resnet import SpectralResNet1D, MultiTaskSpectralResNet1D
from DL.models.unet import SpectralUNet1D
from DL.models.unet_mask import MultiTaskSpectralUNet1D, MultiTaskSpectralUNet1DFusion
from DL.train.losses import LossMSEGradMSE, LossMSEGradMSELogRatio, LossMSEGradMSELaplacienMSE, LossMSEGradSpectral, MultiTaskPSDIntervalLoss, MaskMultiTaskLoss

def train_and_evaluate(model_type="unet", dataset_path="psd_dataset_splits_high_amplitude.npz", save_dir="checkpoints_ha_Laplacian", epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_spectral_{model_type}.pth")

    # 1. Load Data Splits
    train_ds = PSDSplitDataset(filepath=dataset_path, split="train")
    val_ds = PSDSplitDataset(filepath=dataset_path, split="val")
    test_ds = PSDSplitDataset(filepath=dataset_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    if model_type.lower() == "unet":
        model = SpectralUNet1D().to(device)
    elif model_type.lower() == "resnet":
        model = SpectralResNet1D().to(device)
    else:
        raise ValueError("Choose 'unet' or 'resnet'")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = LossMSEGradMSELaplacienMSE(lambda_grad=0.5, lambda_curvature=0.1) #LossMSEGradSpectral(lambda_grad=0.5, lambda_spectral=0.02, cutoff_ratio=0.25) #LossMSEGradMSELaplacienMSE(lambda_grad=0.5, lambda_curvature=0.1) #LossMSEGradMSE(lambda_grad=0.5) # LossMSEGradMSELogRatio(lambda_grad=0.5, lambda_ratio=0.1) 

    # 3. Training Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for noisy_x, clean_y in train_loader:
            noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
            
            optimizer.zero_grad()
            pred_y = model(noisy_x)
            loss = criterion(pred_y, clean_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_x, clean_y in val_loader:
                noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
                pred_y = model(noisy_x)
                loss = criterion(pred_y, clean_y)
                val_loss += loss.item() * noisy_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, save_path)

    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.6f} | Model saved to '{save_path}'")    

    # 4. Final Evaluation on UNSEEN Test Set
    print("\n--- Running Final Evaluation on Test Set ---")
    model.load_state_dict(
        torch.load(save_path, map_location=device)["model_state_dict"]
    )
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for noisy_x, clean_y in test_loader:
            noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
            pred_y = model(noisy_x)
            loss = criterion(pred_y, clean_y)
            test_loss += loss.item() * noisy_x.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"Final Unbiased Test Loss: {test_loss:.6f}")

    return model

# if __name__ == "__main__":
#     trained_model = train_and_evaluate(model_type="resnet", epochs=30)




# ==========================================================================
# Intervals model
# ==========================================================================


# =====================================================================
# 1. TEST & EVALUATION FUNCTION
# =====================================================================
def evaluate_model_metrics(model, test_loader, criterion, device):
    """
    Evaluates a trained multi-task model on the test dataset and computes:
    - Composite Test Loss
    - Peak Count Accuracy (%)
    - Peak Center Frequency (f0) MAE (Hz)
    - Interval Bounds (f_L, f_R) MAE (Hz)
    - Mean Interval Overlap (IoU)
    """
    model.eval()

    test_loss = 0.0
    correct_count_preds = 0
    total_samples = 0
    
    f0_errors = []
    bounds_errors = []
    iou_scores = []

    with torch.no_grad():
        for noisy_x, clean_y, targets, masks, n_peaks in test_loader:
            noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
            targets, masks, n_peaks = targets.to(device), masks.to(device), n_peaks.to(device)

            pred_clean, pred_intervals, pred_count = model(noisy_x)
            loss, _, _, _ = criterion(
                pred_clean, clean_y, pred_intervals, targets, masks, pred_count, n_peaks
            )
            test_loss += loss.item() * noisy_x.size(0)

            # --- 1. Peak Count Accuracy ---
            rounded_pred_count = torch.round(torch.clamp(pred_count, min=0, max=4))
            correct_count_preds += (rounded_pred_count == n_peaks).sum().item()
            total_samples += noisy_x.size(0)

            # --- 2. Frequency Range & Position Metrics ---
            for b in range(noisy_x.size(0)):
                num_true_peaks = int(n_peaks[b].item())
                
                for k in range(num_true_peaks):
                    true_f0, true_l, true_r = targets[b, k].cpu().numpy()
                    pred_f0, pred_l, pred_r = pred_intervals[b, k].cpu().numpy()

                    # Position error (f0)
                    f0_errors.append(abs(pred_f0 - true_f0))

                    # Edge boundary errors (f_left, f_right)
                    bounds_errors.append(abs(pred_l - true_l))
                    bounds_errors.append(abs(pred_r - true_r))

                    # 1D Interval Intersection over Union (IoU)
                    intersection_left = max(true_l, pred_l)
                    intersection_right = min(true_r, pred_r)
                    intersection = max(0.0, intersection_right - intersection_left)

                    union_left = min(true_l, pred_l)
                    union_right = max(true_r, pred_r)
                    union = max(1e-6, union_right - union_left)

                    iou_scores.append(intersection / union)

    # Compute Summaries
    test_loss /= total_samples
    count_accuracy = (correct_count_preds / total_samples) * 100.0
    mean_f0_mae = np.mean(f0_errors) if len(f0_errors) > 0 else 0.0
    mean_bounds_mae = np.mean(bounds_errors) if len(bounds_errors) > 0 else 0.0
    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0

    print("\n" + "=" * 50)
    print("        RUNNING EVALUATION ON TEST SET           ")
    print("=" * 50)
    print(f" * Test Composite Loss           : {test_loss:.6f}")
    print(f" * Peak Count Accuracy           : {count_accuracy:.2f}%")
    print(f" * Peak Center Frequency (f0) MAE: {mean_f0_mae:.3f} Hz")
    print(f" * Interval Bounds MAE (f_L, f_R): {mean_bounds_mae:.3f} Hz")
    print(f" * Mean Interval Overlap (IoU)  : {mean_iou:.3f} (1.0 = Perfect)")
    print("=" * 50 + "\n")

    return {
        "test_loss": test_loss,
        "count_accuracy": count_accuracy,
        "f0_mae": mean_f0_mae,
        "bounds_mae": mean_bounds_mae,
        "mean_iou": mean_iou,
    }


# =====================================================================
# 2. TRAINING FUNCTION (CALLS TEST FUNCTION AT END)
# =====================================================================
def train_multitask_psd_model(
    dataset_path="psd_dataset_intervals_ha.npz", 
    save_dir="checkpoints_multitask",
    epochs=30, 
    batch_size=64, 
    lr=1e-3, 
    f_max=50.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-Task PSD Model on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_multitask_resnet.pth")

    # 1. Dataset & Loaders
    train_ds = PSDIntervalDataset(filepath=dataset_path, split="train")
    val_ds = PSDIntervalDataset(filepath=dataset_path, split="val")
    test_ds = PSDIntervalDataset(filepath=dataset_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model & Loss
    model = MultiTaskSpectralResNet1D(max_peaks=4, f_max=f_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = MultiTaskPSDIntervalLoss()

    best_val_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for noisy_x, clean_y, targets, masks, n_peaks in train_loader:
            noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
            targets, masks, n_peaks = targets.to(device), masks.to(device), n_peaks.to(device)

            optimizer.zero_grad()
            pred_clean, pred_intervals, pred_count = model(noisy_x)
            
            loss, _, _, _ = criterion(
                pred_clean, clean_y, pred_intervals, targets, masks, pred_count, n_peaks
            )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy_x.size(0)

        train_loss /= len(train_ds)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_x, clean_y, targets, masks, n_peaks in val_loader:
                noisy_x, clean_y = noisy_x.to(device), clean_y.to(device)
                targets, masks, n_peaks = targets.to(device), masks.to(device), n_peaks.to(device)

                pred_clean, pred_intervals, pred_count = model(noisy_x)
                loss, _, _, _ = criterion(
                    pred_clean, clean_y, pred_intervals, targets, masks, pred_count, n_peaks
                )
                val_loss += loss.item() * noisy_x.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save Checkpoint for Best Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, save_path)

    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.6f} | Checkpoint saved to '{save_path}'")

    # 4. Load Best Model Checkpoint & Run Evaluation Function
    model.load_state_dict(torch.load(save_path, map_location=device)["model_state_dict"])
    
    test_metrics = evaluate_model_metrics(model, test_loader, criterion, device)

    return model, test_metrics

# if __name__ == "__main__":
#     trained_model = train_multitask_psd_model()


# ==========================================================================
# Masks model
# ==========================================================================

# =====================================================================
# 1. TEST & EVALUATION FUNCTION FOR MASK ARCHITECTURES
# =====================================================================
def evaluate_model_metrics(model, test_loader, criterion, device, center_thresh=0.4, interval_thresh=0.5):
    """
    Evaluates a trained mask-based multi-task model on the test dataset and computes:
    - Composite Test Loss
    - Reconstruction MAE (Log PSD)
    - Peak Center Mask Dice Score
    - Interval Range Mask Dice Score
    - Interval Mask Intersection over Union (IoU)
    """
    model.eval()

    test_loss = 0.0
    total_samples = 0

    reconstruction_maes = []
    center_dice_scores = []
    interval_dice_scores = []
    interval_iou_scores = []

    smooth = 1e-6

    with torch.no_grad():
        for noisy_x, clean_y, target_masks in test_loader:
            noisy_x = noisy_x.to(device)
            clean_y = clean_y.to(device)
            target_masks = target_masks.to(device)  # Shape: (Batch, 2, n_freqs)

            # Model Forward Pass
            pred_clean, pred_masks = model(noisy_x)
            
            # Loss Computation
            loss, _, _ = criterion(pred_clean, clean_y, pred_masks, target_masks)
            test_loss += loss.item() * noisy_x.size(0)
            total_samples += noisy_x.size(0)

            # 1. Reconstruction MAE
            mae = torch.mean(torch.abs(pred_clean - clean_y)).item()
            reconstruction_maes.append(mae)

            # 2. Mask Metrics (Batch-wise processing)
            pred_center = pred_masks[:, 0, :]    # Channel 0
            pred_interval = pred_masks[:, 1, :]  # Channel 1

            target_center = target_masks[:, 0, :]
            target_interval = target_masks[:, 1, :]

            # Binary threshold predictions for discrete metrics evaluation
            bin_pred_center = (pred_center >= center_thresh).float()
            bin_pred_interval = (pred_interval >= interval_thresh).float()

            for b in range(noisy_x.size(0)):
                # Channel 0: Center Mask Dice
                inter_c = (bin_pred_center[b] * target_center[b]).sum().item()
                dice_c = (2.0 * inter_c + smooth) / (bin_pred_center[b].sum().item() + target_center[b].sum().item() + smooth)
                center_dice_scores.append(dice_c)

                # Channel 1: Interval Mask Dice & IoU
                inter_i = (bin_pred_interval[b] * target_interval[b]).sum().item()
                union_i = bin_pred_interval[b].sum().item() + target_interval[b].sum().item() - inter_i

                dice_i = (2.0 * inter_i + smooth) / (bin_pred_interval[b].sum().item() + target_interval[b].sum().item() + smooth)
                iou_i = (inter_i + smooth) / (union_i + smooth)

                interval_dice_scores.append(dice_i)
                interval_iou_scores.append(iou_i)

    # Compute Summaries
    test_loss /= total_samples
    mean_recon_mae = np.mean(reconstruction_maes)
    mean_center_dice = np.mean(center_dice_scores)
    mean_interval_dice = np.mean(interval_dice_scores)
    mean_interval_iou = np.mean(interval_iou_scores)

    print("\n" + "=" * 55)
    print("        RUNNING EVALUATION ON MASK TEST SET        ")
    print("=" * 55)
    print(f" * Test Composite Loss          : {test_loss:.6f}")
    print(f" * Reconstruction Curve MAE     : {mean_recon_mae:.4f}")
    print(f" * Peak Center Mask Dice Score  : {mean_center_dice:.4f} (1.0 = Perfect)")
    print(f" * Interval Range Mask Dice     : {mean_interval_dice:.4f} (1.0 = Perfect)")
    print(f" * Interval Range Mask IoU      : {mean_interval_iou:.4f} (1.0 = Perfect)")
    print("=" * 55 + "\n")

    return {
        "test_loss": test_loss,
        "recon_mae": mean_recon_mae,
        "center_dice": mean_center_dice,
        "interval_dice": mean_interval_dice,
        "interval_iou": mean_interval_iou,
    }


# =====================================================================
# 2. TRAINING FUNCTION FOR MASK ARCHITECTURE
# =====================================================================
def train_mask_psd_model(
    dataset_path="psd_dataset_masks.npz", 
    save_dir="checkpoints_masks",
    epochs=30, 
    batch_size=64, 
    lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Mask Multi-Task PSD Model on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_mask_unet.pth")

    # 1. Dataset & Loaders
    train_ds = PSDMaskDataset(filepath=dataset_path, split="train")
    val_ds = PSDMaskDataset(filepath=dataset_path, split="val")
    test_ds = PSDMaskDataset(filepath=dataset_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model & Loss Function
    model = MultiTaskSpectralUNet1DFusion(in_channels=1, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = MaskMultiTaskLoss(lambda_grad=0.5, lambda_curvature=0.1, lambda_mask=0.1)

    best_val_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for noisy_x, clean_y, target_masks in train_loader:
            noisy_x = noisy_x.to(device)
            clean_y = clean_y.to(device)
            target_masks = target_masks.to(device)  # Shape: (Batch, 2, n_freqs)

            optimizer.zero_grad()
            pred_clean, pred_masks = model(noisy_x)
            
            loss, _, _ = criterion(pred_clean, clean_y, pred_masks, target_masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy_x.size(0)

        train_loss /= len(train_ds)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_x, clean_y, target_masks in val_loader:
                noisy_x = noisy_x.to(device)
                clean_y = clean_y.to(device)
                target_masks = target_masks.to(device)

                pred_clean, pred_masks = model(noisy_x)
                loss, _, _ = criterion(pred_clean, clean_y, pred_masks, target_masks)
                val_loss += loss.item() * noisy_x.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save Checkpoint for Best Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, save_path)

    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.6f} | Checkpoint saved to '{save_path}'")

    # 4. Load Best Model Checkpoint & Run Evaluation Function
    model.load_state_dict(torch.load(save_path, map_location=device)["model_state_dict"])
    
    test_metrics = evaluate_model_metrics(model, test_loader, criterion, device)

    return model, test_metrics


if __name__ == "__main__":
    trained_model, metrics = train_mask_psd_model()


