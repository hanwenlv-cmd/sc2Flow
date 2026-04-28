# %%
import numpy as np
import os
import scanpy as sc
import biolord
from tqdm import tqdm
import warnings


# %%
warnings.simplefilter("ignore", UserWarning)
h5ad_path = '/home/usr/sc2Flow_alb/datasets/sciplex3_pre_2000.h5ad'
adata = sc.read_h5ad(h5ad_path)
adata.obs['drug_dose'] = adata.obs['product_name'].astype(str) + "_" + adata.obs['dose_val'].astype(str)
split_key = 'split_ood_finetuning'


# %%
assert 'rdkit2d_dose' in adata.obsm.keys()
# %%

print(adata)

biolord.Biolord.setup_anndata(
    adata,
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

model = biolord.Biolord(
    adata=adata,
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

if 0:
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
    '''
    Epoch 49/500:  10%|▉         | 49/500 [55:50<8:33:55, 68.37s/it, v_num=1,
    val_generative_mean_accuracy=0.918,
    val_generative_var_accuracy=0.756, val_biolord_metric=0.837,
    val_LOSS_KEYS.RECONSTRUCTION=-2.92, val_LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY=8.33e-11,
    generative_mean_accuracy=0, generative_var_accuracy=0, biolord_metric=0,
    reconstruction_loss=-2.58, unknown_attribute_penalty_loss=11.3]
    Monitored metric val_biolord_metric did not improve in the last 20 records.
    Best score: 0.843. Signaling Trainer to stop.'''
#%%
    model.save('./checkpoints/sciplex/')

else:
    model = biolord.Biolord.load('./checkpoints/sciplex/', adata=adata, device="auto")
# %%
if 0:
    size = 4
    vals = ["generative_mean_accuracy", "generative_var_accuracy", "biolord_metric"]
    fig, axs = plt.subplots(nrows=1, ncols=len(vals), figsize=(size * len(vals), size))

    model.epoch_history = pd.DataFrame().from_dict(model.training_plan.epoch_history)
    for i, val in enumerate(vals):
        sns.lineplot(
            x="epoch",
            y=val,
            hue="mode",
            data=model.epoch_history[model.epoch_history["mode"] == "valid"],
            ax=axs[i],
        )

    plt.tight_layout()
    plt.show()
idx_source = np.where(adata.obs["control"] == 1)[0]
adata_source = adata[idx_source].copy()

ood_drug = adata.obs[split_key] == 'ood'

adata = adata[ood_drug].copy()
preds_dict = model.compute_prediction_adata(
    adata=adata,
    adata_source=adata_source,
    target_attributes=["drug_dose"],
    add_attributes=["cell_type"],
    continuous_attributes='rdkit2d_dose',
)

import torch
from utils.metrics import compute_metrics_fast

result = {}
result_de = {}
for c, v in tqdm(preds_dict.items(), desc='compute metrics'):
    if c[0].split('_')[0] == 'DMSO':
        continue
    if c[0] == 'Alvespimycin (17-DMAG) HCl_0.001': #A549_
        pass
    index = adata.obs['drug_dose'] == c[0]  # c is a tuple
    real = torch.tensor(adata[index].X.toarray()).cuda()
    pred = v.cuda()
    result[c] = compute_metrics_fast(real, pred)

    # use any cell type, cus they have same DEGs
    cell_drug_dose = adata[index].obs['cell_type'].iloc[0]  + '_' + c[0]
    bool_de = adata.var_names.isin(np.array(adata.uns['all_DEGs'][cell_drug_dose]))
    result_de[c] = compute_metrics_fast(real[:,bool_de], pred[:,bool_de])

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

#de
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


result_save_path = f'./results/sciplex/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
print('save to ',result_save_path)
import pickle
with open(result_save_path + 'res' + '.pkl', 'wb') as f:
    pickle.dump(result, f)
with open(result_save_path + 'res_deg' + '.pkl', 'wb') as f:
    pickle.dump(result_de, f)
