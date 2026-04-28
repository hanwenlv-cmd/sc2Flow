import torch
import os
import gc
# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from util.config_tool import get_config
import numpy as np
from dataset import split_data
from model import MLP

from tqdm import tqdm
import scanpy as sc

def split_pack(args, type='train'):
    file_path = args.DataPath
    print(f'reading {file_path}')
    adata = sc.read_h5ad(file_path)
    adata = split_data(adata, args, type)
    return adata

def repeat_n(x, n, device=None):
    """Returns an n-times repeated version of the Tensor x, repetition dimension is axis 0."""
    device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)

def main(args):

    if torch.cuda.is_available():
        device = 'cuda:0'
        print('Using gpu')
    else:
        device = 'cpu'
        print('Using cpu.')

    torch.manual_seed(42)

    batch_size = 512
    dropout = 0.2
    vocab_size = 2
    dims = 2000
    hidden_dim = 768

    if dataset_name == 'zhaoSims':
        n_layers = 12
    elif dataset_name == 'sciplex':
        n_layers = 12
    else:
        n_layers = 8


    adata = split_pack(args,type= 'test')
    gc.collect()

    # probability denoiser model init
    probability_denoiser = MLP(
        input_dim=vocab_size,
        time_dim=768,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        length=dims,
        dropout=dropout,
        condition_num=2).to(device)

    # instantiate a convex path object
    probability_denoiser.load_state_dict(
        torch.load(f'{args.OUTPUT_DIR}/model_best.pt', weights_only=False))

    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

    nfe = 64
    step_size = 1 / nfe

    solver.model.eval()

    if adata.shape[1] != 2000:
        print('select hvgs')
        adata = adata[:, adata.var.highly_variable].copy()
    if hasattr(adata.X, 'toarray'):
        adata.X = (adata.X.toarray() != 0).astype(int)
    else:
        adata.X = (adata.X != 0).astype(int)

    # to 1/0
    binary_data = (adata.X != 0).astype(int)
    adata.X = binary_data

    ctrl_idxs = adata.obs[args.control_key] == 1
    ctrl_adata = adata[ctrl_idxs].copy()
    assert ctrl_adata.obs[args.perturbation_key].unique() == 'DMSO'
    treat_adata = adata[~ctrl_idxs].copy()


    cell_type_list = list(adata.obs[args.cell_type_key].astype('str'))
    cell_types = list(set(cell_type_list))

    args.drug_dose_key = 'drug_dose'
    treat_adata.obs[args.drug_dose_key] = (treat_adata.obs[args.perturbation_key].astype('str') + '_'
                                           + treat_adata.obs[args.dosage_key].astype('str'))

    treated_adata_unique = treat_adata.obs.drop_duplicates(subset=[args.drug_dose_key])

    smiles_list = list(treated_adata_unique[args.SMILES_key])
    dose_list = torch.tensor(treated_adata_unique[args.dosage_key], dtype=torch.float32)
    drug_dose_list = list(treated_adata_unique[args.drug_dose_key])
    exist_cell_drug_dose_list = list(treat_adata.obs[args.cell_drug_dose_key])

    from MoleEmb import extract_mole_embed
    condition_emb = extract_mole_embed(smiles_list).cpu()

    pred_list_all = []
    dose_list_all = []
    smiles_list_all = []
    condition_list_all = []

    for cell_type in tqdm(cell_types):
        src_adata = ctrl_adata[ctrl_adata.obs[args.cell_type_key] == cell_type]
        x_0 = torch.tensor(src_adata.X, dtype=torch.long).to(device)
        for i in range(len(treated_adata_unique)):
            curr_cell_drug_dose = f'{cell_type}_{drug_dose_list[i]}'
            if curr_cell_drug_dose in exist_cell_drug_dose_list:
                c_rep = repeat_n(condition_emb[i],x_0.shape[0]).to(device)
                dose = repeat_n(dose_list[i],x_0.shape[0]).reshape(-1).to(device)
                smiles = [smiles_list[i] for _ in range(x_0.shape[0])]
                condition = [curr_cell_drug_dose for _ in range(x_0.shape[0])]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    n_samples = x_0.shape[0]
                    batch_predictions = []
                    for start_idx in range(0, n_samples, batch_size):
                        end_idx = min(start_idx + batch_size, n_samples)
                        x_0_batch = x_0[start_idx:end_idx]
                        c_rep_batch = c_rep[start_idx:end_idx]
                        dose_batch = dose[start_idx:end_idx]
                        batch_result = solver.sample(x_init=x_0_batch,
                                               step_size=step_size,
                                               verbose=False,
                                               return_intermediates=True,
                                               c_rep=c_rep_batch,
                                               dose=dose_batch,
                                               )
                        batch_predictions.append(batch_result[1].cpu())




                    final_prediction = torch.cat(batch_predictions, dim=0)
                    result = (None, final_prediction)

                pred_list_all.append(result[1])
                dose_list_all.append(dose.cpu())
                smiles_list_all += smiles
                condition_list_all += condition
            else:
                continue
        try:
            dose_tensor= torch.cat(dose_list_all, dim=0)
        except:
            pass
        pred_tensor = torch.cat(pred_list_all, dim=0)

    import anndata as ad
    predicted_adata = ad.AnnData(X = np.array(pred_tensor.cpu()),obs = {'cov_drug_dose_name': condition_list_all,
           'dose_val': dose_tensor.cpu(),
           'SMILES': smiles_list_all,
           'is_predict': True})
    print(predicted_adata)
    predicted_adata.write_h5ad(f'{args.OUTPUT_DIR}/predicted_adata.h5ad')



if __name__ == '__main__':
    for dataset_name in ['sciplex',
        'mcfarland',
        'zhaoSims'
        ]:
        args = get_config(dataset_name)
        k = args.k
        split_key_base = args.split_key
        DATA_PATH = args.DataPath

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                return torch.softmax(self.model(x, t, **extras), dim=-1)

        for i in range(0,k):
            if k != 1:
                args.split_key = split_key_base + '_' + str(i)
            else:
                args.split_key = split_key_base
            args.OUTPUT_DIR = f'./output/{dataset_name}/{args.split_key}'
            os.makedirs(args.OUTPUT_DIR, exist_ok=True)
            if os.path.exists(f'{args.OUTPUT_DIR}/predicted_adata.h5ad'):
                print(f'{args.OUTPUT_DIR}/model_best.pt exists, skip training')
                continue


            main(args)