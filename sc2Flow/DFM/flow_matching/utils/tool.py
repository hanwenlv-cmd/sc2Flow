import torch
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from triton.language import dtype


def ot_matching(x0, x1, maximize=False):
    """
    x0: [Batch, Dim]
    x1: [Batch, Dim]
    maximize: True for Reverse Matching (Max Distance), False for OT-CFM (Min Distance)
    """
    # diff: [B, B, D]
    diff = x0.unsqueeze(1) - x1.unsqueeze(0)
    # cost_matrix: [B, B]
    cost_matrix = torch.sum(diff ** 2, dim=-1)

    if maximize:
        cost_matrix = -cost_matrix


    cost_np = cost_matrix.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)

    x1_aligned = x1[col_idx]

    return x0, x1_aligned

def draw(sample):
    plt.figure()
    plt.bar(np.arange(sample.shape[0]), sample)
    plt.show()

def generate_bimodal_gaussian_onehot(num_samples, vector_length, means, stds):

    samples = np.zeros((num_samples, vector_length), dtype=np.int32)
    positions = np.arange(vector_length)
    combined_prob = np.zeros(vector_length)
    for mean, std in zip(means, stds):
        combined_prob += np.exp(-0.5 * ((positions - mean) / std) ** 2)

    combined_prob = combined_prob / np.max(combined_prob)

    # plt.figure()
    # plt.plot(np.arange(combined_prob.shape[0]), combined_prob)
    # plt.show()

    for i in tqdm(range(num_samples)):

        samples[i] = np.random.binomial(1, combined_prob)

    return samples


class my_dataset(Dataset):
    def __init__(self,src_means,src_stds,trg_means,trg_stds,N,vector_length):
        self.N = N
        self.src = generate_bimodal_gaussian_onehot(
            num_samples=N,
            vector_length=vector_length,
            means=src_means,
            stds=src_stds
        )

        self.trg = generate_bimodal_gaussian_onehot(
            num_samples=N,
            vector_length=vector_length,
            means=trg_means,
            stds=trg_stds
        )

        self.src = torch.tensor(self.src,dtype=torch.long)
        self.trg = torch.tensor(self.trg,dtype=torch.long)


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.src[index], self.trg[index]


class gene_dataset(Dataset):
    def __init__(self,data_path=None):
        adata = sc.read_h5ad(data_path)
        adata = adata[:, adata.var['highly_variable']].copy()
        if hasattr(adata.X, 'toarray'):
            adata.X = (adata.X.toarray() != 0).astype(int)
        else:
            adata.X = (adata.X != 0).astype(int)

        self.dims = adata.shape[1]  # hvg
        self.vocab_size = 2  # express or not


        binary_data = (adata.X!=0).astype(int)
        adata.X = binary_data

        src_index = adata.obs['is_control']
        src_adata = adata[src_index]
        trg_adata = adata[~src_index]

        src_idxs,trg_idxs = self.random_pair_by_celltype(src_adata, trg_adata)
        self.src = torch.tensor(src_adata[src_idxs].X, dtype=torch.long)
        self.trg = torch.tensor(trg_adata[trg_idxs].X, dtype=torch.long)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def random_pair_by_celltype(self, src_adata, trg_adata, cell_type_col='cell_type', seed=42):
        np.random.seed(seed)

        src_types = set(src_adata.obs[cell_type_col].unique())
        trg_types = set(trg_adata.obs[cell_type_col].unique())
        common_types = src_types.intersection(trg_types)

        paired_src_indices = []
        paired_trg_indices = []

        print(f"paring... {len(common_types)} cell types in total.")

        for ct in common_types:
            src_idxs = src_adata.obs[src_adata.obs[cell_type_col] == ct].index.values
            trg_idxs = trg_adata.obs[trg_adata.obs[cell_type_col] == ct].index.values

            n_pairs = max(len(src_idxs), len(trg_idxs))

            replace_src = n_pairs > len(src_idxs)
            replace_trg = n_pairs > len(trg_idxs)

            chosen_src = np.random.choice(src_idxs, n_pairs, replace=replace_src)
            chosen_trg = np.random.choice(trg_idxs, n_pairs, replace=replace_trg)

            paired_src_indices.extend(chosen_src)
            paired_trg_indices.extend(chosen_trg)

        print(f"finished pairing. {len(paired_src_indices)} paris in total.")

        return paired_src_indices, paired_trg_indices