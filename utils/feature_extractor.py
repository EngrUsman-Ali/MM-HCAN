import torch
from models.temporal_model import TemporalFeatureExtractor
from models.spectral_model import SpectralFeatureExtractor

def extract_features(temporal_model, spectral_model, dataloader, device):
    temporal_model.eval()
    spectral_model.eval()
    
    all_temporal_features = []
    all_spectral_features = []
    all_labels = []
    
    with torch.no_grad():
        for temporal_input, spectral_input, labels in dataloader:
            temporal_input = temporal_input.to(device)
            spectral_input = spectral_input.to(device)
            
            temp_features = temporal_model(temporal_input.unsqueeze(1))
            
            spec_features = spectral_model(spectral_input)
            
            all_temporal_features.append(temp_features.cpu())
            all_spectral_features.append(spec_features.cpu())
            all_labels.append(labels)
    
    all_temporal_features = torch.cat(all_temporal_features, dim=0)
    all_spectral_features = torch.cat(all_spectral_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_temporal_features, all_spectral_features, all_labels