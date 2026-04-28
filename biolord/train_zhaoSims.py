
import numpy as np

import scanpy as sc
import biolord
from tqdm import tqdm
import os

import warnings
import torch
from utils.metrics import compute_metrics_fast

def limit(x, n):
    if x.shape[0] > n:
        indices = torch.randperm(x.shape[0])[:n]
        sampled_tensor = torch.index_select(x, 0, indices)
        return sampled_tensor
    else:
        return x
limit_n = 20000
# %%
warnings.simplefilter("ignore", UserWarning)
h5ad_path = '/home/usr/sc2Flow_alb/datasets/zhaoSims2021_pre.h5ad'
original_adata = sc.read_h5ad(h5ad_path)
original_adata.obs['drug_dose'] = original_adata.obs['condition'].astype(str) + "_" + original_adata.obs['dose_val'].astype(str)

# %%
print(original_adata)

biolord.Biolord.setup_anndata(
    original_adata,
    ordered_attributes_keys=["rdkit2d_dose"],
    categorical_attributes_keys=["cell_type"],
)

module_params = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "seed": 42,
}

for split_id in range(3,4):
    split_key = f'split_{split_id}'
    print(f'train dataset on split: {split_key}')
    print(f'test drugs: {original_adata.obs.loc[original_adata.obs[split_key]=="od", "condition"].value_counts()}')
    checkpoint_save_path = f'./checkpoints/zhaoSims/{split_key}'
    result_save_path = f'./results/zhaoSims/{split_key}/'
    train = False

    model = biolord.Biolord(
        adata=original_adata,
        n_latent=32,
        model_name="spatio_temporal_infected",
        module_params=module_params,
        train_classifiers=False,
        split_key=split_key,
    )

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }

    if train:
        model.train(
            max_epochs=500,
            batch_size=512,
            plan_kwargs=trainer_params,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=10,
            num_workers=1,
            enable_checkpointing=False,
        )
#%%
        model.save(checkpoint_save_path)
    else:
        adata_temp = original_adata.copy()
        adata_temp.X = 0
        model = biolord.Biolord.load(checkpoint_save_path, adata=adata_temp, device="auto")
# %%
    adata = original_adata.copy()
    idx_source = np.where(adata.obs["control"] == 1)[0]
    adata_source_all = adata[idx_source].copy()
    ood_drug = adata.obs[split_key] == 'ood'
    adata = adata[ood_drug].copy()
    cell_drug_dose_set = set(adata.obs['cov_drug_dose_name'])

    pred_dict_all = {}
    cell_types = np.unique(adata.obs['cell_type'])
    for cell_type in tqdm(cell_types):
        adata_source = adata_source_all[adata_source_all.obs['cell_type'] == cell_type]
        preds_dict = model.compute_prediction_adata(
            adata=adata,
            adata_source=adata_source,
            target_attributes=["drug_dose"],
            add_attributes=["cell_type"],
            continuous_attributes='rdkit2d_dose',
        )
        for k, v in preds_dict.items():
            cell_drug_dose = f'{cell_type}_{k[0]}'
            if cell_drug_dose in cell_drug_dose_set:
                pred_dict_all[cell_drug_dose] = v
            else:
                continue

    result = {}

    sum = 0
    for _, v in pred_dict_all.items():
        sum += v.shape[0]

    result_de = {}
    for c, v in tqdm(pred_dict_all.items(), desc='compute metrics'):
        if c.split('_')[1] == 'DMSO':
            continue
        index = adata.obs['cov_drug_dose_name'] == c
        real = torch.tensor(adata[index].X.toarray()).cuda()
        pred = v.cuda()

        real = limit(real.cpu(), limit_n).cuda()
        pred = limit(pred.cpu(), limit_n).cuda()

        result[c] = compute_metrics_fast(real, pred)
        # use any cell type, cus they have same DEGs
        bool_de = adata.var_names.isin(np.array(adata.uns['all_DEGs'][c]))
        result_de[c] = compute_metrics_fast(real[:, bool_de], pred[:, bool_de])

    test_logs = {}
    for c, metrics in result.items():
        for m, v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(v)

    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]
        print(metric, ': ', test_logs[metric])

    # de
    test_de_logs = {}
    for c, metrics in result_de.items():
        for m, v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_de_logs:
                test_de_logs[key] = []
            test_de_logs[key].append(v)

    for metric in test_de_logs.keys():
        test_de_logs[metric] = [torch.tensor(test_de_logs[metric]).mean()]
        print(metric, ' de: ', test_de_logs[metric])

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    print('save to ', result_save_path)
    import pickle

    with open(result_save_path + 'res' + '.pkl', 'wb') as f:
        pickle.dump(result, f)
    with open(result_save_path + 'res_deg' + '.pkl', 'wb') as f:
        pickle.dump(result_de, f)
