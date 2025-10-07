import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_hypergraph(features, threshold=0.7):
    features = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features
    N = features.shape[0]

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(features)

    # Build incidence matrix H
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and sim_matrix[i, j] >= threshold:
                H[i, j] = 1  # Connect node i to j if similarity is above threshold

    # Node and edge degrees
    node_degree = np.sum(H, axis=1)
    edge_degree = np.sum(H, axis=0)

    # Inverse square roots for normalization
    inv_sqrt_D_node = np.diag(1.0 / np.sqrt(node_degree + 1e-8))
    inv_D_edge = np.diag(1.0 / (edge_degree + 1e-8))  # Use inverse, not inverse square root

    # Compute hypergraph Laplacian correctly
    L = np.eye(N) - inv_sqrt_D_node @ H @ inv_D_edge @ H.T @ inv_sqrt_D_node

    return torch.tensor(L, dtype=torch.float32), torch.tensor(H, dtype=torch.float32)