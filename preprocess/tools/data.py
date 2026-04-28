import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import scanpy as sc


def K_fold_split(adata, group_name, k, trg_keys):

    col_name = group_name
    n_splits = k

    unique_types = np.array(adata.obs[col_name].unique())
    unique_types = unique_types[unique_types != trg_keys['control_perturbation_name']]
    assert k <= len(unique_types)


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(unique_types)):
        split_key = f'split_{fold_id}'
        adata.obs[split_key] = trg_keys['train']

        test_types = unique_types[test_idx]

        print(f"Fold {fold_id}: Test Types -> {test_types}")


        adata.obs.loc[adata.obs[col_name].isin(test_types), split_key] = trg_keys['test']

        train_mask = adata.obs[split_key] == 'train'

        valid_num = len(train_mask)//50
        selected_train_indices = adata.obs[train_mask].sample(n=valid_num).index

        adata.obs.loc[selected_train_indices, split_key] = trg_keys['valid']

        print(adata.obs[split_key].value_counts())
    return adata

def calculate_DEGs(adata,trg_keys,full=False,n_genes=50,save_path=None):
    de_genes = {}
    de_genes_quick = {}

    adata_df = adata.to_df()

    adata_df = adata_df.join(adata.obs[trg_keys['perturbation_name']])  # Ensures correct alignment
    dmso = adata_df[adata_df[trg_keys['perturbation_name']] == trg_keys['control_perturbation_name']].mean(numeric_only=True)

    if full:
        for cond, df in tqdm(adata_df.groupby(trg_keys['perturbation_name'])):
            if cond != trg_keys['control_perturbation_name']:
                drug_mean = df.mean(numeric_only=True)
                de_50_idx = np.argsort(abs(drug_mean - dmso))[-n_genes:]
                de_genes_quick[cond] = drug_mean.index[de_50_idx].values

        de_genes = de_genes_quick
    else:
        sc.tl.rank_genes_groups(
            adata,
            groupby=trg_keys['perturbation_name'],
            reference=trg_keys['control_perturbation_name'],
            rankby_abs=True,
            n_genes=n_genes
        )
        for cond in tqdm(np.unique(adata.obs[trg_keys['cov_drug_dose_name']])):
            drug_name = cond.split('_')[1]
            if drug_name != trg_keys['control_perturbation_name']:
                df = sc.get.rank_genes_groups_df(adata, group=drug_name)
                de_genes[cond] = df['names'][:n_genes].values
    adata.uns[trg_keys['DEGs']] = de_genes
    del adata.uns['rank_genes_groups']

    if save_path is not None:
        adata.write_h5ad(save_path)
        print(f"adata saved to {save_path}")

    return adata

    if adata.shape[0] != 2000:
        print('save 2000 hvgs only')
        adata = adata[:, adata.var.highly_variable].copy()
        adata.write_h5ad('/home/usr/sc2Flow_lab/datasets/sciplex3_pre_2000.h5ad')

        return adata

def get_smiles_by_CID(CID_list):
    from chembl_webresource_client.new_client import new_client
    molecule = new_client.molecule
    id2smiles = {}
    for chembl_id in tqdm(CID_list):
        results = molecule.filter(chembl_id=chembl_id).only(['molecule_structures'])

        if results:

            structures = results[0]['molecule_structures']

            if structures and 'canonical_smiles' in structures:
                smiles = structures['canonical_smiles']
                id2smiles[chembl_id] =  smiles
            else:
                print(f"{chembl_id}' smiles not found")
        else:
            print(f"ChEMBL ID: {chembl_id} not found")

    return id2smiles