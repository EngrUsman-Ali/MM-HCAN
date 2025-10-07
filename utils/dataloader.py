
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision import transforms
from dataset import TemporalSpectralDataset


def get_dataloaders(root_dir, batch_size=32):
    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),       # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load Dataset
    full_dataset = TemporalSpectralDataset(root_dir, transform=transform)

    # 🔁 Shuffle the entire dataset before splitting
    indices = torch.randperm(len(full_dataset)).tolist()
    full_dataset = torch.utils.data.Subset(full_dataset, indices)

    # Split into Train (80%) and Test (20%)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Further split Train into Fine-tune (60%) and Validation (20%)
    fine_tune_size = int(0.75 * train_size)  # 60% of 80% = 60%
    val_size = train_size - fine_tune_size
    fine_tune_dataset, val_dataset = random_split(train_dataset, [fine_tune_size, val_size])

    # Create Dataloaders
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return fine_tune_loader, val_loader, test_loader


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

    unique_labels = sorted(set(all_labels))  # or use torch.unique if labels are tensors
    print(f"{loader_name} - Temporal: {total_temporal}, Spectral: {total_spectral}, Labels: {total_labels}")
    print(f"{loader_name} - Unique Labels: {unique_labels}")

if __name__ == "__main__":
    root_dir = "Dataset"  # replace with your actual dataset path
    fine_tune_loader, val_loader, test_loader = get_dataloaders(root_dir, batch_size=1)

    count_loader_samples_and_labels(fine_tune_loader, "Fine-tune Loader")
    count_loader_samples_and_labels(val_loader, "Validation Loader")
    count_loader_samples_and_labels(test_loader, "Test Loader")