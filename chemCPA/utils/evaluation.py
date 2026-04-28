import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from utils.metrics import compute_metrics_fast
import matplotlib.pyplot as plt

def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    """
    return np.where(x)[0]
    #return np.flatnonzero(x)

def repeat_n(x, n):
    """
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)

def evaluate(model, dataset, genes_control):
    res_dict = {}
    res_deg_dict = {}

    n_rows = genes_control.size(0)
    genes_control = genes_control.to(model.device)

    # Determine the key format by checking the first key in de_genes
    if len(dataset.de_genes) > 0:
        sample_key = next(iter(dataset.de_genes))
        key_parts = len(sample_key.split('_'))
        use_cell_drug_only = (key_parts == 2)  # True if keys are cell_drug format
    else:
        use_cell_drug_only = True  # Default to cell_drug if no keys exist

    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    unique_categories, category_counts = np.unique(dataset.pert_categories, return_counts=True)

    total_combinations = 0
    missing_combinations = 0
    missing_keys = set()

    for cell_drug_dose_comb, category_count in tqdm(list(zip(unique_categories, category_counts)),
                                                    desc='evaluating'):
        if dataset.perturbation_key is None:
            break

        # if category_count<5:
        #     continue

        if "dmso" in cell_drug_dose_comb.lower() or "control" in cell_drug_dose_comb.lower():
            #continue
            # 'DMSO' has a SMILES representation, so we can use it
            pass

        # Get the appropriate key based on de_genes format
        if use_cell_drug_only:
            lookup_key = '_'.join(cell_drug_dose_comb.split('_')[:2])  # Just cell_drug
        else:
            lookup_key = cell_drug_dose_comb  # Full cell_drug_dose

        total_combinations += 1

        if lookup_key not in dataset.de_genes:
            missing_combinations += 1
            missing_keys.add(lookup_key)
            assert f'{lookup_key} not in deg list'
            #continue

        if(lookup_key not in dataset.de_genes.keys()):
            print(f'{lookup_key} not found in de_genes.keys')
            continue
        bool_de = dataset.var_names.isin(np.array(dataset.de_genes[lookup_key]))

        # need at least two genes to be able to calc r2 score
        if len(bool_de) < 2:
            continue

        #bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        bool_category = pert_categories_index.isin([cell_drug_dose_comb])
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]

        emb_covs = [repeat_n(cov[idx], n_rows) for cov in dataset.covariates]
        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_rows).squeeze(),
                repeat_n(dataset.dosages[idx], n_rows).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_rows)

        genes_pred = model.predict(
            genes=genes_control,
            drugs_idx=emb_drugs[0],
            dosages=emb_drugs[1],
            covariates=emb_covs,
        )[0].detach()



        dim = genes_control.size(1)
        mean_pred = genes_pred[:, :dim] # default set (use mean as final prediction)

        y_true = dataset.genes[idx_all, :].to(device="cuda")

        def limit(x, n):
            if x.shape[0] > n:
                indices = torch.randperm(x.shape[0])[:n]
                sampled_tensor = torch.index_select(x, 0, indices)
                return sampled_tensor
            else:
                return x

        y_true = limit(y_true.cpu(), 20000).to(device="cuda")
        mean_pred = limit(mean_pred.cpu(), 20000).to(device="cuda")


        res = compute_metrics_fast(y_true, mean_pred)
        res_de = compute_metrics_fast(y_true[:,bool_de], mean_pred[:,bool_de])

        res_dict[cell_drug_dose_comb] = res
        res_deg_dict[cell_drug_dose_comb] = res_de

    if missing_combinations > 0:
        print(f"\nWarning: Missing DE genes for {missing_combinations} out of {total_combinations} combinations")
        print("Example missing combinations:", list(missing_keys)[:3])

    return res_dict, res_deg_dict