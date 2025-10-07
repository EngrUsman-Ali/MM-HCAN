import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from utils.dataloader import get_dataloaders
from utils.feature_extractor import extract_features
from models.temporal_model import TemporalFeatureExtractor
from models.spectral_model import SpectralFeatureExtractor

class TemporalClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = TemporalFeatureExtractor() 
        self.head     = nn.Linear(512, num_classes)   

    def forward(self, x):
        feats = self.backbone(x)      
        return self.head(feats)    


def train_and_save_best_model(root_dir, epochs=100, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    temporal_model = TemporalClassifier().to(device)
    spectral_model = SpectralFeatureExtractor().to(device)
    
    optimizer_temp = optim.Adam(temporal_model.parameters(), lr=1e-4)
    optimizer_spec = optim.Adam(spectral_model.parameters(), lr=1e-4)
    
    criterion = CrossEntropyLoss()
    
    fine_tune_loader, val_loader, _ = get_dataloaders(root_dir, batch_size)
    
    best_acc_temp = 0
    best_acc_spec = 0

    for epoch in range(epochs):

        spectral_model.train()
        total_loss_spec = 0
        total_correct_spec = 0
        total_samples_spec = 0

        for _, x_spec, y in fine_tune_loader:
            x_spec, y = x_spec.to(device), y.to(device)
            optimizer_spec.zero_grad()
            out = spectral_model(x_spec)
            loss = criterion(out, y)
            loss.backward()
            optimizer_spec.step()

            total_loss_spec += loss.item() * y.size(0)
            
            preds = torch.argmax(out, dim=1)
            total_correct_spec += (preds == y).sum().item()
            total_samples_spec += y.size(0)

        avg_loss_spec = total_loss_spec / total_samples_spec
        train_acc_spec = total_correct_spec / total_samples_spec

        acc_spec = validate(spectral_model, val_loader, device, modality='spectral')
        print(f"[Spectral] Epoch {epoch} | "
            f"Train Loss: {avg_loss_spec:.4f}, "
            f"Train Acc: {train_acc_spec:.4f}, "
            f"Val Acc: {acc_spec:.4f}")

        if acc_spec > best_acc_spec:
            print(f"Saving Best Spectral at accuracy {acc_spec}")
            best_acc_spec = acc_spec
            torch.save(spectral_model.state_dict(), "best_spectral.pth")
    
        temporal_model.train()
        total_loss_temp = 0
        total_correct_temp = 0
        total_samples_temp = 0
        
        for x_temp, _, y in fine_tune_loader:
            x_temp, y = x_temp.to(device), y.to(device)
            optimizer_temp.zero_grad()
            out = temporal_model(x_temp)
            loss = criterion(out, y)
            loss.backward()
            optimizer_temp.step()

            total_loss_temp += loss.item() * y.size(0)
            
            preds = torch.argmax(out, dim=1)
            total_correct_temp += (preds == y).sum().item()
            total_samples_temp += y.size(0)

        avg_loss_temp = total_loss_temp / total_samples_temp
        train_acc_temp = total_correct_temp / total_samples_temp

        acc_temp = validate(temporal_model, val_loader, device, modality='temporal')
        print(f"[Temporal] Epoch {epoch} | "
            f"Train Loss: {avg_loss_temp:.4f}, "
            f"Train Acc: {train_acc_temp:.4f}, "
            f"Val Acc: {acc_temp:.4f}")

        if acc_temp > best_acc_temp:
            print(f"Saving Best Temporal at accuracy {acc_temp}")
            best_acc_temp = acc_temp
            torch.save(temporal_model.backbone.state_dict(), "best_temporal.pth")


def validate(model, loader, device, modality='temporal'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_temp, x_spec, y in loader:
            if modality == 'temporal':
                x = x_temp.to(device)
            elif modality == 'spectral':
                x = x_spec.to(device)
            else:
                raise ValueError("Invalid modality. Use 'temporal' or 'spectral'.")
            
            y = y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    root_dir = "Dataset"
    train_and_save_best_model(root_dir)