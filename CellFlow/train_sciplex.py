import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import functools
import scanpy as sc
import cellflow
from cellflow.model import CellFlow
from cellflow.utils import match_linear
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca, annotate_compounds, get_molecular_fingerprints


path = '/home/usr/sc2Flow_lab/data/sciplex3_pre_2000.h5ad'
adata = sc.read_h5ad(path)

if adata.shape[1] != 2000:
    print('select genes number to 2000')
    adata = adata[:, adata.var.highly_variable].copy()
    assert adata.shape[1] == 2000

adata_train = adata[adata.obs["mode"] == "train"]
adata_test = adata[(adata.obs["mode"] == "test") | (adata.obs["condition"]=="control")]

assert len(set(adata_test.obs['condition'])) == 9 + 1
#centered_pca(adata_train, method="rapids", keep_centered_data=False, n_comps=100)
#centered_pca(adata_train, keep_centered_data=False, n_comps=100)

print('calculating pca...')
sc.tl.pca(adata_train, n_comps=100)

adata_train.varm["X_mean"] = adata_train.var["means"].values if "means" in adata_train.var.keys() else adata_train.X.mean(axis=0).A1 if hasattr(adata_train.X, 'A1') else adata_train.X.mean(axis=0)

adata_train.varm["PCs"] = adata_train.varm["PCs"]
project_pca(adata_test, ref_adata=adata_train)
print('done')


cf = CellFlow(adata_train, solver="otfm")
cf.prepare_data(
    sample_rep = "X_pca",
    control_key = "is_control",
    perturbation_covariates = {"drug_perturbation": ("condition" ,)},
    perturbation_covariate_reps = {"drug_perturbation": "fingerprints"},
    max_combination_length = 1,
    null_value = 0.0,
)

cf.prepare_validation_data(
    adata_train,
    name="train",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)

cf.prepare_validation_data(
    adata_test,
    name="test",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)

layers_before_pool = {
    "drug_perturbation": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
}

layers_after_pool = {
    "layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0,
}

match_fn = functools.partial(match_linear, epsilon=1.0, tau_a=1.0, tau_b=1.0)

cf.prepare_model(
    condition_mode="deterministic",
    regularization=0.0,
    pooling="mean",
    layers_before_pool=layers_before_pool,
    layers_after_pool=layers_after_pool,
    condition_embedding_dim=64,
    cond_output_dropout=0.9,
    hidden_dims=[2048, 2048, 2048],
    conditioning="concatenation",
    decoder_dims=[4096, 4096, 4096],
    probability_path={"constant_noise": 1.5},
    match_fn=match_fn,
    linear_projection_before_concatenation=True,
    )
train = 1
if train:
    metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
    decoded_metrics_callback = cellflow.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared"])
    callbacks = [metrics_callback, decoded_metrics_callback]

    num_iterations = 500000

    cf.train(
            num_iterations=num_iterations,
            batch_size=1024,
            callbacks=callbacks,
            valid_freq=num_iterations+1,
        )
    cf.save(dir_path='./',file_prefix=f'{num_iterations}', overwrite=True)


