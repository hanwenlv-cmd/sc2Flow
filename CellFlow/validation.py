
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import numpy as np
import pandas as pd

from tqdm import tqdm
import anndata as ad
import scanpy as sc
from cellflow.model import CellFlow
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca, annotate_compounds, get_molecular_fingerprints

#%%
path = '/home/usr/sc2Flow_lab/datasets/sciplex3_pre_2000.h5ad'
adata = sc.read_h5ad(path)
#%%
adata_test = adata[(adata.obs["mode"] == "test") | (adata.obs["condition"]=="control")].copy()

assert len(adata_test.obs["condition"].unique()) == 9+1
print('calculating pca...')
sc.tl.pca(adata, n_comps=100)

adata.varm["X_mean"] = adata.var["means"].values if "means" in adata.var.keys() else adata.X.mean(axis=0).A1 \
    if hasattr(adata.X, 'A1') else adata.X.mean(axis=0)

adata.varm["PCs"] = adata.varm["PCs"]
project_pca(adata_test, ref_adata=adata)
print('done')

adata_ctrl_for_prediction = adata_test[adata_test.obs["is_control"].to_numpy()]
covariate_data = adata_test[~adata_test.obs["is_control"].to_numpy()].obs.drop_duplicates(subset=["drug_dose"])
print(covariate_data)
#%%



adata_temp = ad.AnnData()
cf = CellFlow(adata_temp,"otfm")
path = '/home/usr/sc2Flow_lab/datasets/CellFlow/500000_CellFlow.pkl'
model = cf.load(path)

predict = False

if predict:
    preds = model.predict(adata=adata_ctrl_for_prediction,
                          sample_rep="X_aligned",
                          condition_id_key="drug_dose",
                          covariate_data=covariate_data)

    with open('preds.pkl', 'wb') as f:
        pickle.dump(preds, f)
else:
    with open('preds.pkl', 'rb') as f:
        preds = pickle.load(f)

#%%
adata_preds = []
for cond, array in preds.items():
    obs_data = pd.DataFrame({
        'drug_dose': [cond] * array.shape[0]
    })
    adata_pred = ad.AnnData(X=np.empty((len(array), model.adata.n_vars)), obs=obs_data)
    adata_pred.obsm["X_pca"] = np.asarray(np.squeeze(array))
    adata_preds.append(adata_pred)

adata_preds = ad.concat(adata_preds)
adata_preds.var_names = model.adata.var_names


reconstruct_pca(adata_preds, use_rep="X_pca", ref_adata=model.adata)
adata_preds.X = adata_preds.layers["X_recon"]

import torch
from utils.metrics_torch import compute_metrics_fast
result = {}
result_de = {}

for c, _ in tqdm(preds.items(), desc='compute metrics'):
    drug = c.split('_')[-2]
    dose = c.split('_')[-1]
    drug_dose = drug + '_' + str(float(dose)/10000)


    index = adata_test.obs['drug_dose'] == c  # c is a tuple
    real = torch.tensor(adata_test[index].X.toarray(),dtype=torch.float32).cuda()
    pred = torch.tensor(adata_preds[adata_preds.obs['drug_dose'] == c].X,dtype=torch.float32).cuda()
    result[drug_dose] = compute_metrics_fast(real, pred)



    c_deg = adata_test[adata_test.obs['drug_dose'] == c].obs['cov_drug_dose_name'][0]

    bool_de = adata.var_names.isin(np.array(adata.uns['all_DEGs'][c_deg]))
    result_de[drug_dose] = compute_metrics_fast(real[:, bool_de], pred[:, bool_de])
    #calcuate mean
    # save test_logs
base_dir = '/home/usr/sc2Flow/CellFlow/output/result/sciplex/mode'
os.makedirs(base_dir, exist_ok=True)
with open(f'{base_dir}/res.pkl', 'wb') as f:
    pickle.dump(result, f)
with open(f'{base_dir}/res_de.pkl', 'wb') as f:
    pickle.dump(result_de, f)

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

test_logs_de = {}
for c, metrics in result_de.items():
    for m, v in metrics.items():
        key = f'test_{m}_mean_de'
        if key not in test_logs_de:
            test_logs_de[key] = []
        test_logs_de[key].append(v)

for metric in test_logs_de.keys():
    test_logs_de[metric] = [torch.tensor(test_logs_de[metric]).mean()]
    print(metric, ': ', test_logs_de[metric])

with open('test_logs.pkl', 'wb') as f:
    pickle.dump(test_logs, f)
