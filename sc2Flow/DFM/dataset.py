from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm

def process_data(args, type='train'):
    file_path = args.DataPath
    print(f'reading {file_path}')
    adata = sc.read_h5ad(file_path)

    adata = split_data(adata, args, type)
    dataset = discrete_dataset(adata, args)
    return dataset


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
        adata = adata[ctrl_idxs | index].copy()
        print(f'test drugs: {adata.obs[args.perturbation_key].unique()}')
    return adata


class discrete_dataset(Dataset):
    def __init__(self, adata,args):
        if adata.shape[1] != 2000:
            print('select hvgs')
            adata = adata[:, adata.var.highly_variable].copy()
        if hasattr(adata.X, 'toarray'):
            adata.X = (adata.X.toarray() != 0).astype(int)
        else:
            adata.X = (adata.X != 0).astype(int)

        # to 1/0
        binary_data = (adata.X!=0).astype(int)
        adata.X = binary_data

        # transfer cell type string to number so we can use index to find cell group instead of string hash
        # especially when there are many cell types
        cell_types = np.sort(adata.obs[args.cell_type_key].unique())
        cell_type2id = {cell_type: i for i, cell_type in enumerate(cell_types)}

        ctrl_idxs = adata.obs[args.control_key]==1
        ctrl_adata = adata[ctrl_idxs].copy()
        assert ctrl_adata.obs[args.perturbation_key].unique() == 'DMSO'
        treat_adata = adata[~ctrl_idxs].copy()
        self.treat_cell_id = list(treat_adata.obs[args.cell_type_key].map(cell_type2id).values)

        #group ctrl samples' index by cell type
        self.ctrl_idxs_grouped = []
        for cell_type in tqdm(cell_types):
            idxs = np.where(ctrl_adata.obs[args.cell_type_key]==cell_type)[0]
            assert len(idxs) != 0
            #print(f'{cell_type} does not have control sample, allocated random control samples')
            self.ctrl_idxs_grouped.append(idxs)

        self.src = torch.tensor(ctrl_adata.X, dtype=torch.long)
        self.trg = torch.tensor(treat_adata.X, dtype=torch.long)

        # condition representation
        smile_list = list(treat_adata.obs[args.SMILES_key].astype('str'))
        # we don' generate control samples in this stage, so only treated data's drug embs are needed
        self.smile_list = smile_list
        smiles_list_unique = np.sort(list(set(smile_list)))
        smile2idx = {smiles: i for i, smiles in enumerate(smiles_list_unique)}
        self.sample_idx2emb_idx = [smile2idx[smile] for i, smile in enumerate(smile_list)]

        from MoleEmb import extract_mole_embed
        self.condition_emb = extract_mole_embed(smiles_list_unique).cpu()

        # sample information
        self.condition_list = list(
            treat_adata.obs[args.cell_type_key].astype('str') + '_' \
            + treat_adata.obs[args.perturbation_key].astype('str') + '_' \
            + treat_adata.obs[args.dosage_key].astype('str')
        )

        # drug dosage
        self.dose = torch.tensor(treat_adata.obs[args.dosage_key],dtype=torch.float32)
        self.rng = np.random.RandomState(42)
    def __len__(self):
        return len(self.trg)

    def __getitem__(self, index):
        # sample from ctrl
        cell_id = self.treat_cell_id[index]
        sampled_idx = self.rng.choice(self.ctrl_idxs_grouped[cell_id])

        return  self.src[sampled_idx], \
                self.trg[index], \
                {
                "c_rep":self.condition_emb[self.sample_idx2emb_idx[index]],
                "dose": self.dose[index],
                "condition": self.condition_list[index],
                "smiles": self.smile_list[index],
                }

