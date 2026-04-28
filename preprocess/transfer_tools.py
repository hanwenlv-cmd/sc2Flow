import pandas as pd
from scipy.stats import rdist

from tools.data import K_fold_split, calculate_DEGs, get_smiles_by_CID
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os

trg_keys={
        'control_label':'control', # 0/1
        'control_perturbation_name':'DMSO',
        'cell_type':'cell_type',
        'perturbation_name':'condition',
        'dosage':'dose_val', # 0~1
        'dosage_max_value':10000., #uM
        'cov_drug_dose_name':'cov_drug_dose_name',
        'SMILES':'SMILES',
        'train': 'train',
        'valid': 'test',
        'test': 'ood',
        'DEGs': 'all_DEGs',
    }


def to_chemCPA_data_format(adata, src_keys, save_path=None,n_hvgs=2000,rdkit_emb_path=None):
    if os.path.exists(save_path):
        adata = sc.read_h5ad(save_path)
        if 'rdkit2d_dose' not in adata.obsm.keys():
            print('create rdkit2d_dose')
            assert rdkit_emb_path is not None
            rdkit_emb_path = rdkit_emb_path
            from tools.RDkit_tool.helper import canonicalize_smiles
            try:
                df = pd.read_parquet(rdkit_emb_path)
            except FileNotFoundError:
                print(f'{rdkit_emb_path} not found')
                exit()
            emb_list = []
            for i in tqdm(range(adata.shape[0]), desc='create rdkit2d_dose to .obsm'):
                smiles = adata.obs[trg_keys['SMILES']].iloc[i]
                dose = adata.obs[trg_keys['dosage']].iloc[i]
                emb_list.append(
                    np.concatenate([df.loc[canonicalize_smiles(smiles)].values, [dose]])
                )
            rdkit2d_dose = np.array(emb_list)

            adata.obsm['rdkit2d_dose'] = rdkit2d_dose
            adata.write_h5ad(save_path)
            print(f"adata saved to {save_path}")
    else:
        adata.obs[trg_keys['cell_type']] = adata.obs[src_keys['cell_type']]

        adata.obs[trg_keys['perturbation_name']] = adata.obs[src_keys['perturbation_name']]
        adata.obs[trg_keys['perturbation_name']] = adata.obs[trg_keys['perturbation_name']].apply(
            lambda x: trg_keys['control_perturbation_name'] if x == src_keys['control_perturbation_name'] else x
        )

        assert isinstance(src_keys['dosage_max_value'], float)
        adata.obs[trg_keys['dosage']] = adata.obs[src_keys['dosage']].astype(float) / src_keys['dosage_max_value']
        adata.obs.loc[adata.obs[trg_keys['perturbation_name']] == trg_keys['control_perturbation_name'], trg_keys['dosage']] = 0.

        adata.obs[trg_keys['cov_drug_dose_name']] = adata.obs[trg_keys['cell_type']].astype(str) + "_" \
                                          + adata.obs[trg_keys['perturbation_name']].astype(str) + "_" \
                                          + adata.obs[trg_keys['dosage']].astype(str)

        if src_keys['control_label']=='':
            adata.obs[trg_keys['control_label']] = (
                    adata.obs[trg_keys['perturbation_name']] == trg_keys['control_perturbation_name']).astype(int)
        else:
            adata.obs[trg_keys['control_label']] = adata.obs[src_keys['control_label']]

        #split data:
        unique_types = np.array(adata.obs[trg_keys['perturbation_name']].unique())
        unique_types = unique_types[unique_types != trg_keys['control_perturbation_name']]
        k = min(5, len(unique_types))
        print(f'split dataset into {k} folds')
        adata = K_fold_split(adata=adata,group_name=trg_keys['perturbation_name'],k=k,trg_keys=trg_keys).copy()

        if src_keys['SMILES'] == '':
            print(f"write SMILES {save_path}")
            #adata.obs[src_keys['cid']] = adata.obs[src_keys['cid']].astype(str)
            adata.obs.loc[adata.obs[trg_keys['control_label']]==1, src_keys['cid']] = 'CHEMBL504'
            cid = adata.obs[src_keys['cid']]
            cid_list = list(set(cid))
            assert 'nan' not in cid_list
            cid2smiles = get_smiles_by_CID(cid_list)
            adata.obs[trg_keys['SMILES']] = adata.obs[src_keys['cid']].apply(lambda x: cid2smiles[x])
        else:
            adata.obs[trg_keys['SMILES']] = adata.obs[src_keys['SMILES']]

        if adata.X.shape[0]!=n_hvgs:
            print(f'select {n_hvgs} hvgs')
            if sp.issparse(adata.X):
                X_flat = adata.X.toarray()[0].flatten()
            else:
                X_flat = adata.X[0].flatten()

            if np.all(np.equal(np.mod(X_flat, 1), 0)):
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_hvgs)
            adata = adata[:, adata.var.highly_variable].copy()
        if trg_keys['DEGs'] not in adata.uns:
            print(f"calculate DEGs {save_path}")
            adata = calculate_DEGs(adata,trg_keys=trg_keys,save_path=save_path).copy()

    print(adata.obs)


    return adata




if __name__ == '__main__':
    import scanpy as sc

    dataset_path = {
        'McFarlandTsherniak2020': './mcfarland_2020.h5ad',
        'ZhaoSims2021': './zhaoSims2021.h5ad',
    }

    dataset_name = 'McFarlandTsherniak2020'

    if dataset_name == 'ZhaoSims2021':
        path = dataset_path['ZhaoSims2021']
        save_path = path.replace('.h5ad', '_pre.h5ad')
        rdkit_emb_path = '/home/usr/sc2Flow_lab/dataset_factory/tools/RDkit_tool/zhaoSims_rdkit_emb_skip.parquet'
        adata = sc.read_h5ad(path)
        adata.obs['chembl-ID'] = adata.obs['chembl-ID'].astype(str).copy()
        condition = (adata.obs['perturbation'] != 'control') & (adata.obs['chembl-ID']=='nan')
        adata = adata[~condition, :].copy()

        # fill in '' if key doesn't exist in source dataset
        obs_source_keys = {
            'control_label': '',
            'control_perturbation_name': 'control',
            'cell_type': 'tissue', #unknown
            'perturbation_name': 'perturbation',
            'dosage': 'dose_value',
            'dosage_max_value': 50., #'0.2', '2.5', '50' um
            'SMILES': '',
            'cid': 'chembl-ID',
        }
        to_chemCPA_data_format(adata, obs_source_keys, save_path,2000,rdkit_emb_path)


    if dataset_name == 'McFarlandTsherniak2020':
        path = dataset_path['McFarlandTsherniak2020']
        save_path = path.replace('.h5ad', '_pre.h5ad')
        rdkit_emb_path = '/home/usr/sc2Flow_lab/dataset_factory/tools/RDkit_tool/mcfarland_rdkit_emb_skip.parquet'
        adata = sc.read_h5ad(path)
        adata.obs['chembl-ID'] = adata.obs['chembl-ID'].astype(str).copy()
        condition = (adata.obs['perturbation'] != 'control') & (adata.obs['chembl-ID']=='nan')
        adata = adata[~condition, :].copy()

        # fill in '' if key doesn't exist in source dataset
        obs_source_keys = {
            'control_label': '',
            'control_perturbation_name': 'control',
            'cell_type': 'cell_line',
            'perturbation_name': 'perturbation',
            'dosage': 'dose_value',
            'dosage_max_value': 10.,
            'SMILES': '',
            'cid': 'chembl-ID',
        }
        to_chemCPA_data_format(adata, obs_source_keys, save_path,2000,rdkit_emb_path)
