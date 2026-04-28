from collections.abc import Sequence
import torch
import numpy as np
from geomloss import SamplesLoss
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel

__all__ = [
    "compute_metrics",
    "compute_metrics_fast",
    "compute_mean_metrics",
    "compute_scalar_mmd",
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance_fast",
    "maximum_mean_discrepancy",
]


def compute_r_squared(x, y) -> float:
    """Compute the R squared score between means of the true (x) and predicted (y) distributions.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the R squared score.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    x = np.asarray(x)
    y = np.asarray(y)
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-2) -> torch.Tensor:
    """
    Compute the Sinkhorn divergence between x and y using geomloss.

    Parameters
    ----------
    x : torch.Tensor
        An array of shape [num_samples, num_features].
    y : torch.Tensor
        An array of shape [num_samples, num_features].
    epsilon : float
        The regularization parameter (entropic regularization strength).

    Returns
    -------
    torch.Tensor
        A scalar tensor denoting the Sinkhorn divergence value.
    """
    # geomloss expects [batch, N, D] or [N, D]; we assume uniform weights
    # Sinkhorn divergence = S_ε(x, y) - 0.5 * S_ε(x, x) - 0.5 * S_ε(y, y)
    # But SamplesLoss with `loss="sinkhorn"` and `debias=True` does exactly that!
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,                     # p=2 for squared Euclidean (matching SqEuclidean)
        blur=epsilon ** 0.5,    # blur = sqrt(ε), because geomloss uses blur^p = ε
        debias=True,            # enables Sinkhorn *divergence* (debiased); if False → Sinkhorn distance
        backend="tensorized",         # auto-selects online/tensorized
    )
    # Assume uniform weights (1/n)
    # geomloss auto-normalizes if no weights provided
    div = sinkhorn(x, y)
    return div


def compute_e_distance_fast(x: torch.Tensor, y: torch.Tensor) -> float:
    # try:
    #     import pykeops
    # except ImportError:
    #     raise ImportError("Please install pykeops to use this function.")

    gl_default = SamplesLoss(loss="energy", blur=0.05)
    loss = gl_default(x, y)
    return loss

def manual_energy_loss(X, Y):
    """
    E[|X-Y|] - 0.5*E[|X-X'|] - 0.5*E[|Y-Y'|]
    """

    D_XY = torch.cdist(X, Y, p=2)  # Shape: (N, M)
    D_XX = torch.cdist(X, X, p=2)  # Shape: (N, N)
    D_YY = torch.cdist(Y, Y, p=2)  # Shape: (M, M)

    E_XY = D_XY.mean()
    E_XX = D_XX.mean()
    E_YY = D_YY.mean()

    loss = E_XY - 0.5 * E_XX - 0.5 * E_YY

    return loss

def compute_metrics(x, y) -> dict[str, float]:
    """Compute a set of metrics between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A dictionary containing the following computed metrics:

        - the r squared score.
        - the sinkhorn divergence with ``epsilon = 1.0``.
        - the sinkhorn divergence with ``epsilon = 10.0``.
        - the sinkhorn divergence with ``epsilon = 100.0``.
        - the energy distance value.
        - the mean maximum discrepancy loss
    """
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance_fast(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics

def compute_mean_metrics(metrics: dict[str, dict[str, float]], prefix: str = "") -> dict[str, list[float]]:
    """Compute the mean value of different metrics.

    Parameters
    ----------
        metrics
            A dictionary where the keys indicate the name of the pertubations and the values are
            dictionaries containing computed metrics.
        prefix
            A string definining the prefix of all metrics in the output dictionary.

    Returns
    -------
        A dictionary where the keys indicate the metrics and the values contain the average metric
        values over all pertubations.
    """
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict: dict[str, list[float]] = {prefix + met_name: [] for met_name in metric_names}
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict

def compute_scalar_mmd(x, y, gammas=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """

    Parameters
    ----------
    x, y : np.ndarray or torch.Tensor
    gammas : list[float]
    device : str
    Returns
    -------
    float : mean MMD
    """

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    x = x.to(device)
    y = y.to(device)

    mmd_values = []

    for gamma in gammas:
        blur = 1.0 / np.sqrt(gamma)
        loss_fn = SamplesLoss(loss="gaussian", blur=blur, backend="auto")
        loss_val = loss_fn(x, y).item()
        mmd_values.append(2 * loss_val)

    return np.mean(mmd_values)

@torch.jit.script
def rbf_kernel_fast(x, y, gamma: float) -> torch.Tensor:
    xx = (x**2).sum(1)
    yy = (y**2).sum(1)
    xy = x @ y.T
    sq_distances = xx[:, None] + yy - 2 * xy
    return torch.exp(-gamma * sq_distances)

def maximum_mean_discrepancy(x, y, gamma: float = 1.0, exact: bool = False) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gamma
            Parameter for the rbf kernel.
        exact
            Use exact or fast rbf kernel.

    Returns
    -------
        A scalar denoting the squared maximum mean discrepancy loss.
    """
    kernel = rbf_kernel if exact else rbf_kernel_fast
    xx = kernel(x, x, gamma)
    xy = kernel(x, y, gamma)
    yy = kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def manual_scalar_mmd(x, y, gammas: Sequence[float] | None = None) -> float:
    """Compute the Mean Maximum Discrepancy (MMD) across different length scales

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))


def compute_metrics_fast(x, y) -> dict[str, float]:
    """Compute metrics which are fast to compute

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A dictionary containing the following computed metrics:

        - the r squared score.
        - the energy distance value.
        - the mean maximum discrepancy loss
    """
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)

    try:
        metrics["e_distance"] = compute_e_distance_fast(x, y)
    except:
        print('cuda might out of memory, Switched to cpu')
        metrics["e_distance"] = manual_energy_loss(x.cpu(), y.cpu())

    try:
        metrics["mmd"] = compute_scalar_mmd(x, y)
    except:
        print('cuda might out of memory, Switched to cpu')
        metrics["mmd"] = manual_scalar_mmd(x.cpu(), y.cpu())
    return metrics
