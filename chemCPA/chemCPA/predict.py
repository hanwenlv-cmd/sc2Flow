import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
import torch
import hydra
from lightning_module import ChemCPA
from data.perturbation_data_module import PerturbationDataModule
from data.data import load_dataset_splits
from tqdm import tqdm
from utils.evaluation import evaluate
import os
import pickle

@hydra.main(version_base=None, config_path="../config/", config_name="lincs")
def main(args):
    if hasattr(main, 'split'):
        split = main.split
        dataset_name = main.dataset_name
    else:
        raise ValueError('Please specify split')

    data_params = args["dataset"]
    data_params["split_key"] = split

    hash_path = f'/{data_params["split_key"]}'
    print(f'test {dataset_name} on data split {data_params["split_key"]}')

    checkpoint_path = args['training']['save_dir'] + '/' + dataset_name + hash_path +'/last.ckpt'
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    state_dict = checkpoint['state_dict']

    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)
    dm = PerturbationDataModule(datasplits=datasets, train_bs=args["model"]["hparams"]["batch_size"])
    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    dataset.debug_print()

    model = ChemCPA(args, dataset_config)
    filtered_state_dict = {}
    for name, param in state_dict.items():
        if not any(skip_layer in name for skip_layer in ['adversary_covariates', 'covariates_embeddings']):
            filtered_state_dict[name] = param

    model.load_state_dict(filtered_state_dict,strict=False)
    model = model.cuda()
    model = model.eval()
    dm.setup(stage='predict')

    genes_control = dm.ood_control_dataset.genes

    res_dict,res_deg_dict = evaluate(model.model,dm.ood_treated_dataset,genes_control)

    test_logs = {}
    for c, metrics in res_dict.items():
        for m, v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(v)

    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]
        print(metric, ': ', test_logs[metric])

    # deg
    test_deg_logs = {}
    for c, metrics in res_deg_dict.items():
        for m, v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_deg_logs:
                test_deg_logs[key] = []
            test_deg_logs[key].append(v)

    for metric in test_deg_logs.keys():
        test_deg_logs[metric] = [torch.tensor(test_deg_logs[metric]).mean()]
        print(metric, ': ', test_deg_logs[metric])

    result_save_path = f'./results/{dataset_name}/{data_params["split_key"]}/'
    print('save to ',result_save_path)
    os.makedirs(result_save_path, exist_ok=True)

    with open(result_save_path + 'res' + '.pkl', 'wb') as f:
        pickle.dump(res_dict, f)
    with open(result_save_path + 'res_deg' + '.pkl', 'wb') as f:
        pickle.dump(res_deg_dict, f)

if __name__ == "__main__":
    import sys
    dataset_config = {
        'sciplex':1,
        'zhaoSims':4,
        'mcfarland':5,
    }

    dataset_name = 'sciplex'
    # dataset_name = 'mcfarland'
    # zhaoSims

    k = dataset_config[dataset_name]
    i = 2
    if k==1:
        split = 'split_ood_finetuning'
    else:
        split = 'split_' + str(i)

    main.split= split
    main.dataset_name = dataset_name
    sys.argv = [
        "predict.py",
        f"--config-name", f"{dataset_name}_{2000}_genes"
    ]
    main()

