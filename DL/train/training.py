import os
import torch
from torch.utils.data import DataLoader
from DL.data.generate_synthetic_data import PSDSplitDataset  
from DL.models.resnet import SpectralResNet1D
from DL.models.unet import SpectralUNet1D
from DL.train.losses import LossMSEGradMSE, LossMSEGradMSELogRatio

def train_and_evaluate(model_type="unet", dataset_path="psd_dataset_splits_high_amplitude.npz", save_dir="checkpoints_ha", epochs=20, batch_size=64, lr=1e-3):
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
    criterion = LossMSEGradMSE(lambda_grad=0.5) # LossMSEGradMSELogRatio(lambda_grad=0.5, lambda_ratio=0.1) 

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

if __name__ == "__main__":
    trained_model = train_and_evaluate(model_type="resnet", epochs=15)