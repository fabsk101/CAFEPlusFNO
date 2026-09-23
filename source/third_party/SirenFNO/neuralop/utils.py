from typing import Callable, List, Optional, Tuple, Union
from math import prod
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )

def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2*n_params
    return n_params


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


@torch.no_grad()
def evaluate_rel_l2_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    rollout: int,
    rollout_fn: Callable[[torch.nn.Module, torch.Tensor, int, bool], torch.Tensor],
    step_rel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    full_rel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
    step_count_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], int]] = None,
) -> Tuple[float, float]:
    model.eval()
    total_step_sum = 0.0
    total_full_sum = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        batch_size = xb.size(0)
        total_samples += batch_size

        pred = rollout_fn(model, xb, rollout, pushforward_detach=True)
        steps = (
            step_count_fn(pred, yb, rollout) if step_count_fn is not None else rollout
        )

        for t in range(steps):
            total_step_sum += _as_float(step_rel_fn(pred[:, t], yb[:, t]))

        total_full_sum += _as_float(full_rel_fn(pred, yb))

    step_avg_rel_l2 = total_step_sum / (total_samples * rollout)
    full_rel_l2 = total_full_sum / total_samples
    return step_avg_rel_l2, full_rel_l2


def rollout_step_model_1d_scalar(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    steps: int,
    pushforward_detach: bool = True,
) -> torch.Tensor:
    """Autoregressive rollout for 1D scalar outputs.

    x_hist: [B, C_in, X] -> returns [B, steps, X]
    """
    state = x_hist
    preds = []
    for _ in range(steps):
        y1 = model(state)  # [B, 1, X]
        preds.append(y1)
        y_in = y1.detach() if pushforward_detach else y1
        if state.shape[1] == 1:
            state = y_in
        else:
            state = torch.cat([state[:, 1:, :], y_in], dim=1)
    pred_seq = torch.cat(preds, dim=1)  # [B, steps, X]
    return pred_seq


def rollout_step_model_1d_channel_first(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    steps: int,
    pushforward_detach: bool = True,
    field_dim: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive rollout using channel-first tensors (1D spatial grid)."""
    state = x_hist
    preds = []
    for _ in range(steps):
        y1 = model(state)  # [B, field_dim, X]
        preds.append(y1.unsqueeze(1))
        y_in = y1.detach() if pushforward_detach else y1
        step_field_dim = field_dim if field_dim is not None else y1.shape[1]
        if state.shape[1] == step_field_dim:
            state = y_in
        else:
            state = torch.cat([state[:, step_field_dim:, :], y_in], dim=1)
    pred_seq = torch.cat(preds, dim=1)  # [B, steps, field_dim, X]
    return pred_seq


def rollout_step_model_nd_channel_first(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    steps: int,
    pushforward_detach: bool = True,
    field_dim: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive rollout using channel-first tensors (ND spatial grid)."""
    state = x_hist
    preds = []
    for _ in range(steps):
        y1 = model(state)
        preds.append(y1.unsqueeze(1))
        y_in = y1.detach() if pushforward_detach else y1
        step_field_dim = field_dim if field_dim is not None else y1.shape[1]
        if state.shape[1] == step_field_dim:
            state = y_in
        else:
            state = torch.cat([state[:, step_field_dim:, ...], y_in], dim=1)
    pred_seq = torch.cat(preds, dim=1)
    return pred_seq


def rel_l2_sum_over_batch(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    B = x.size(0)
    diff = (x - y).reshape(B, -1)
    tgt = y.reshape(B, -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)  # [B]
    den = torch.linalg.vector_norm(tgt, ord=2, dim=1)  # [B]
    return (num / (den + eps)).sum()


def rollout_loss_rel_l2(pred_seq: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
    """Sum over time of relative L2 over the batch."""
    T = y_seq.shape[1]
    loss = 0.0
    for t in range(T):
        loss = loss + rel_l2_sum_over_batch(pred_seq[:, t], y_seq[:, t])
    return loss


def rollout_loss_lp(
    pred_seq: torch.Tensor,
    y_seq: torch.Tensor,
    lp_rel_train: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    require_lp: bool = False,
) -> torch.Tensor:
    """Channel-wise relative L2 summed over time (LpLoss-based)."""
    if require_lp and lp_rel_train is None:
        raise RuntimeError("LpLoss for training has not been initialised.")
    loss = 0.0
    for t in range(y_seq.shape[1]):
        loss = loss + lp_rel_train(pred_seq[:, t], y_seq[:, t])
    return loss


def full_rel_l2_rel(pred_seq: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
    B = pred_seq.size(0)
    return rel_l2_sum_over_batch(pred_seq.reshape(B, -1), y_seq.reshape(B, -1))


def full_rel_l2_lp(
    pred_seq: torch.Tensor,
    y_seq: torch.Tensor,
    lp_rel_train: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    B = pred_seq.size(0)
    spatial_shape = pred_seq.shape[3:]
    pred_full = pred_seq.reshape(B, -1, *spatial_shape)
    y_seq_full = y_seq.reshape(B, -1, *spatial_shape)
    return lp_rel_train(pred_full, y_seq_full)


def make_lp_losses(spatial_dims: int):
    from neuralop.losses import LpLoss

    lp_rel_train = LpLoss(d=spatial_dims, p=2, reduction="mean")
    lp_rel_eval_sum = LpLoss(d=spatial_dims, p=2, reduction="mean")
    return lp_rel_train, lp_rel_eval_sum


@torch.no_grad()
def evaluate_rel_l2_mean(
    model: torch.nn.Module,
    loader: DataLoader,
    rollout: int,
    rollout_fn: Callable[[torch.nn.Module, torch.Tensor, int, bool], torch.Tensor],
    device: Union[str, torch.device],
    lp_rel_eval_sum: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    require_lp: bool = False,
) -> float:
    if require_lp and lp_rel_eval_sum is None:
        raise RuntimeError("LpLoss for evaluation has not been initialised.")

    model.eval()
    total = 0.0
    denom = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        B, T = yb.size(0), yb.size(1)
        pred = rollout_fn(model, xb, rollout, pushforward_detach=True)
        total += lp_rel_eval_sum(pred, yb).item()
        denom += B * T
    return total / max(1, denom)


def wandb_login(api_key_file="../config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT. 
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``). 
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular 
        coefficient is the coefficient's location along the diagonal, indexed 
        from the top-left corner of the 2d FFT output. 
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum


Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None

def compute_rank(tensor):
    # Compute the matrix rank of a tensor
    rank = torch.matrix_rank(tensor)
    return rank

def compute_stable_rank(tensor):
    # Compute the stable rank of a tensor
    tensor = tensor.detach()
    fro_norm = torch.linalg.norm(tensor, ord='fro')**2
    l2_norm = torch.linalg.norm(tensor, ord=2)**2
    rank = fro_norm / l2_norm
    rank = rank
    return rank

def compute_explained_variance(frequency_max, s):
    # Compute the explained variance based on frequency_max and singular
    # values (s)
    s_current = s.clone()
    s_current[frequency_max:] = 0
    return 1 - torch.var(s - s_current) / torch.var(s)

def get_project_root():
    root = Path(__file__).parent.parent
    return root
