from torch.utils.data import Dataset
import numpy as np
import scipy.sparse
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import scanpy as sc

def process_data(args, type='train',DFM_output_path=None):
    file_path = args.data_path
    print(f'reading {file_path}')
    adata = sc.read_h5ad(file_path)

    adata = split_data(adata, args, type)
    dataset = gene_dataset(adata, args, DFM_output_path=DFM_output_path)
    return dataset


def list2vector(ids,values):
    vector = np.zeros(2000)
    vector[ids-1] = values
    from utils import draw
    draw(vector)
    return  vector

def split_data(adata, args, type='train'):
    control_key = args.control_key
    split_key = args.split_key
    test_key = args.test_key

    ctrl_idxs = adata.obs[control_key]
    if type == 'train':
        index = adata.obs[split_key] == 'train'
        adata = adata[ctrl_idxs | index].copy()
    elif type == 'test':
        index = adata.obs[split_key] == test_key
        adata = adata[index].copy()
        print(f'test drugs: {adata.obs[args.perturbation_key].unique()}')
    return adata

class gene_dataset(Dataset):
    def __init__(self, adata, args, DFM_output_path=None):
        SMILES_key = args.SMILES_key
        perturbation_key = args.perturbation_key
        dosage_key = args.dosage_key
        cell_type_key = args.cell_type_key

        if adata.shape[1] != 2000:
            adata = adata[:, adata.var.highly_variable].copy()

        self.ids, self.values = self.create_tokens_data(adata)

        smile_list = list(adata.obs[SMILES_key].astype('str'))
        smiles_list_unique = np.sort(list(set(smile_list)))
        smile2idx = {smiles: i for i, smiles in enumerate(smiles_list_unique)}
        self.sample_idx2emb_idx = [smile2idx[smile] for i, smile in enumerate(smile_list)]

        from MoleEmb import extract_mole_embed
        self.condition_emb = extract_mole_embed(smiles_list_unique).cpu()

        if DFM_output_path is not None:
            self.condition_list = adata.obs['cov_drug_dose_name']
        else:
            self.condition_list = list(
                adata.obs[cell_type_key].astype('str') + '_'
                + adata.obs[perturbation_key].astype('str') + '_'
                + adata.obs[dosage_key].astype('str')
            )

        self.dose = torch.tensor(adata.obs[dosage_key],dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
            return self.ids[index], \
                self.values[index], \
                self.condition_emb[self.sample_idx2emb_idx[index]], \
                self.dose[index], \
                self.condition_list[index]


    def collate_fn(self, batch):

        g_id = [item[0] for item in batch]
        g_exp = [item[1] for item in batch]
        c_rep = [item[2] for item in batch]
        dose = [item[3] for item in batch]
        condition = [item[4] for item in batch]

        g_id = pad_sequence(g_id, batch_first=True, padding_value=0)
        g_exp = pad_sequence(g_exp, batch_first=True, padding_value=0)

        return g_id, g_exp, torch.stack(c_rep), torch.stack(dose), condition

    def create_tokens_data(self, adata):
        if scipy.sparse.issparse(adata.X):
            X_csr = adata.X.tocsr()
        else:
            X_csr = scipy.sparse.csr_matrix(adata.X)

        rows = X_csr.shape[0]
        ids, values = [], []

        for i in tqdm(range(rows)):
            start_ptr = X_csr.indptr[i]
            end_ptr = X_csr.indptr[i + 1]

            indices = X_csr.indices[start_ptr:end_ptr] + 1
            value = X_csr.data[start_ptr:end_ptr]
            value = 2 * (value / 6.) - 1.

            id = torch.tensor(indices, dtype=torch.long)
            value = torch.tensor(value)
            ids.append(id)
            values.append(value)

        return ids, values