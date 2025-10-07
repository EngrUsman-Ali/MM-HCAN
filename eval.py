import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models.temporal_model import TemporalFeatureExtractor
from models.spectral_model import SpectralFeatureExtractor
from models.hypergraph_builder import build_hypergraph
from models.hgnn_with_triplet import HGNNLayer
from models.multihead_attention import SingleModalityFusion
from models.final_classifier import FinalClassifier

from utils.dataloader import get_dataloaders
from utils.feature_extractor import extract_features 
import os
import numpy as np


def load_best_models(device):
    temporal_model = TemporalFeatureExtractor().to(device)
    temporal_model.load_state_dict(torch.load("checkpoints/temporal.pth"))
    temporal_model.eval()
    spectral_model = SpectralFeatureExtractor().to(device)
    spectral_model.load_state_dict(torch.load("checkpoints/spectral.pth"))
    spectral_model.eval()
    return temporal_model, spectral_model


def build_all_hypergraphs(temp_features, spec_features, concat_features, k=5):
    L_temp, _ = build_hypergraph(temp_features, threshold=0.8)
    L_spec, _ = build_hypergraph(spec_features, threshold=0.8)
    L_concat, _ = build_hypergraph(concat_features, threshold=0.8)
    return {
        'L_temp': L_temp,
        'L_spec': L_spec,
        'L_concat': L_concat
    }


def evaluate_mm_hcan(loader, temporal_model, spectral_model,
                     hg_layer_concat, fusion_layer, classifier, device):

    hg_layer_concat.eval()
    fusion_layer.eval()
    classifier.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion_cls = CrossEntropyLoss()

    with torch.no_grad():
        for x_temp, x_spec, y in loader:
            x_temp, x_spec, y = x_temp.to(device), x_spec.to(device), y.to(device)

            temp_feat = temporal_model(x_temp)
            spec_feat = spectral_model(x_spec)
            fused_feat = temp_feat + spec_feat

            laplacians = build_all_hypergraphs(temp_feat, spec_feat, fused_feat, k=5)
            L_temp = laplacians['L_temp'].to(device)
            L_spec = laplacians['L_spec'].to(device)
            L_concat = laplacians['L_concat'].to(device)

            updated_concat = hg_layer_concat(fused_feat, L_concat)

            fused = fusion_layer(updated_concat)

            logits = classifier(fused)
            preds = torch.argmax(logits, dim=1)

            loss = criterion_cls(logits, y)
            total_loss += loss.item() * y.size(0)

            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def load_best_val_model(hg_layer_concat, fusion_layer, classifier, device):
    
    checkpoint = torch.load("checkpoints/HGNN.pth", map_location=device)

    hg_layer_concat.load_state_dict(checkpoint['hg_layers']['concat'])

    fusion_layer.load_state_dict(checkpoint['fusion'])

    classifier.load_state_dict(checkpoint['classifier'])

    hg_layer_concat.to(device)
    fusion_layer.to(device)
    classifier.to(device)

    hg_layer_concat.eval()
    fusion_layer.eval()
    classifier.eval()

    print("✅ Best validation model loaded.")
    return hg_layer_concat, fusion_layer, classifier


def test_best_val_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, _, test_loader = get_dataloaders("Dataset", batch_size=16)
    print("✅ Test dataloader created")

    temporal_model, spectral_model = load_best_models(device)

    hg_layer_concat = HGNNLayer(512, 256, 512).to(device)
    fusion_layer = SingleModalityFusion(dim=512).to(device)
    classifier = FinalClassifier(input_dim=512).to(device)

    hg_layer_concat, fusion_layer, classifier = load_best_val_model(
        hg_layer_concat, fusion_layer, classifier, device
    )

    print("🔬 Evaluating best val model on test set...")
    test_loss, test_acc = evaluate_mm_hcan(
        test_loader, temporal_model, spectral_model,
        hg_layer_concat, fusion_layer, classifier, device
    )
    print(f"🧪 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    test_best_val_model()