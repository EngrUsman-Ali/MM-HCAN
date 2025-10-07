import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models.temporal_model import TemporalFeatureExtractor
from models.spectral_model import SpectralFeatureExtractor
from utils.feature_extractor import extract_features
from models.hypergraph_builder import build_hypergraph
from models.hgnn_with_triplet import HGNNLayer, TripletLoss
from models.multihead_attention import MultiHeadAttentionFusion,SingleModalityFusion
from models.final_classifier import FinalClassifier
from utils.dataloader import get_dataloaders
import os
import numpy as np

def load_best_models(device):
    temporal_model = TemporalFeatureExtractor().to(device)
    temporal_model.load_state_dict(torch.load("best_temporal.pth"))
    temporal_model.eval()

    spectral_model = SpectralFeatureExtractor().to(device)
    spectral_model.load_state_dict(torch.load("best_spectral.pth"))
    spectral_model.eval()

    return temporal_model, spectral_model


def build_all_hypergraphs(temp_features, spec_features, concat_features,k):
    L_temp, _ = build_hypergraph(temp_features, threshold=0.85)
    L_spec, _ = build_hypergraph(spec_features, threshold=0.85)
    L_concat, _ = build_hypergraph(concat_features, threshold=0.85)

    return {
        'L_temp': L_temp,
        'L_spec': L_spec,
        'L_concat': L_concat
    }

import numpy as np

def select_triplets(features, labels, device='cuda'):

    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

    anchors = []
    positives = []
    negatives = []

    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        if len(idxs) < 2:
            continue
        np.random.shuffle(idxs)
        anchor_idx = idxs[0]
        positive_idx = idxs[1]

        neg_labels = unique_labels[unique_labels != label]
        if len(neg_labels) == 0:
            continue
        neg_label = np.random.choice(neg_labels)
        neg_idxs = np.where(labels == neg_label)[0]
        if len(neg_idxs) == 0:
            continue 

        neg_idx = np.random.choice(neg_idxs)

        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(neg_idx)

    if not anchors:
        return None, None, None 
    return (
        features[anchors].to(device),
        features[positives].to(device),
        features[negatives].to(device)
    )


def train_mm_hcan_with_triplet(
        temporal_model,spectral_model,
    ft_loader, hg_layer_concat,
    fusion_layer, classifier, device, epoch_idx,optimizer
):

    criterion_cls = CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.7)

    hg_layer_concat.train()
    fusion_layer.train()
    classifier.train()

    total_loss_total = 0
    total_correct = 0
    total_samples = 0

    for x_temp, x_spec, y in ft_loader:
        x_temp, x_spec, y = x_temp.to(device), x_spec.to(device), y.to(device)

        current_labels = y.unique().cpu().numel()
        optimizer.zero_grad()

        with torch.no_grad():
            temp_feat = temporal_model(x_temp)
            spec_feat = spectral_model(x_spec)
            fused_feat = temp_feat + spec_feat

        laplacians = build_all_hypergraphs(temp_feat, spec_feat, fused_feat,k=5)

        L_spec = laplacians['L_spec'].to(device)
        L_concat = laplacians['L_concat'].to(device)

        updated_concat = hg_layer_concat(fused_feat, L_concat)

        anchor_c, positive_c, negative_c = select_triplets(updated_concat, y, device=device)

        if anchor_c is not None:
            loss_triplet = criterion_triplet(anchor_c, positive_c, negative_c)
        else:
            print(f"Triplet Loss is None")
            loss_triplet = torch.tensor(0.0, device=device)

        fused = fusion_layer(updated_concat)

        logits = classifier(fused)
        loss_cls = criterion_cls(logits, y)

        total_loss = loss_cls + 0.7 * loss_triplet
        total_loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        batch_acc = correct / y.size(0)

        total_correct += correct
        total_samples += y.size(0)
        total_loss_total += total_loss.item() * y.size(0)


    avg_epoch_loss = total_loss_total / total_samples
    epoch_acc = total_correct / total_samples

    return avg_epoch_loss, epoch_acc


def evaluate_mm_hcan(loader, temporal_model, spectral_model,
                    hg_layer_concat,
                     fusion_layer, classifier, device):

 
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
            current_labels = y.unique().cpu().numel()

            temp_feat = temporal_model(x_temp)
            spec_feat = spectral_model(x_spec)
            fused_feat = temp_feat + spec_feat

            laplacians = build_all_hypergraphs(temp_feat, spec_feat, fused_feat,k=5)
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


def count_loader_samples_and_labels(loader, loader_name):
    total_temporal = 0
    total_spectral = 0
    total_labels = 0
    all_labels = []

    for temporal, spectral, label in loader:
        total_temporal += temporal.size(0)
        total_spectral += spectral.size(0)
        total_labels += label.size(0)
        all_labels.extend(label.tolist())

    unique_labels = sorted(set(all_labels))
    print(f"{loader_name} - Temporal: {total_temporal}, Spectral: {total_spectral}, Labels: {total_labels}")
    print(f"{loader_name} - Unique Labels: {unique_labels}")

def run_full_pipeline(root_dir="Dataset", epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("checkpoints", exist_ok=True)

    ft_loader, val_loader, test_loader = get_dataloaders(root_dir, 16)

    count_loader_samples_and_labels(ft_loader, "Fine-tune Loader")
    count_loader_samples_and_labels(val_loader, "Validation Loader")
    count_loader_samples_and_labels(test_loader, "Test Loader")
    print("✅ Dataloaders created")

    temporal_model, spectral_model = load_best_models(device)
    print("✅ Best models loaded")

    hg_layer_concat = HGNNLayer(512, 256, 512).to(device)

    fusion_layer = SingleModalityFusion(dim=512).to(device)
    classifier = FinalClassifier(input_dim=512).to(device)

    optimizer = torch.optim.Adam(
        list(hg_layer_concat.parameters()) +
        list(fusion_layer.parameters()) +
        list(classifier.parameters()),
        lr=1e-4
    )

    best_train_acc = 0.0
    best_val_acc = 0.0

    print("🧠 Training MM-HCAN...")
    for epoch in range(epochs):
        train_loss, train_acc = train_mm_hcan_with_triplet(
            temporal_model,spectral_model, ft_loader, hg_layer_concat,
            fusion_layer, classifier, device, epoch,optimizer
        )
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate_mm_hcan(
            val_loader, temporal_model, spectral_model,
            hg_layer_concat, fusion_layer, classifier, device
        )
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'hg_layers': {
                    'concat': hg_layer_concat.state_dict()
                },
                'fusion': fusion_layer.state_dict(),
                'classifier': classifier.state_dict()
            }, "checkpoints/best_train_model.pth")
            print(f"💾 Saved best train model (Acc: {train_acc:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'hg_layers': {
                 
                    'concat': hg_layer_concat.state_dict()
                },
                'fusion': fusion_layer.state_dict(),
                'classifier': classifier.state_dict()
            }, "checkpoints/best_val_model.pth")
            print(f"💾 Saved best val model (Acc: {val_acc:.4f})")

    print("\n🔬 Evaluating on Test Set...")
    test_loss, test_acc = evaluate_mm_hcan(
        test_loader, temporal_model, spectral_model,
        hg_layer_concat,
        fusion_layer, classifier, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")

    return {
        "train_loss": train_loss,
        "train_acc": best_train_acc,
        "val_loss": val_loss,
        "val_acc": best_val_acc,
        "test_acc": test_acc
    }

if __name__ == "__main__":
    run_full_pipeline(epochs=50)