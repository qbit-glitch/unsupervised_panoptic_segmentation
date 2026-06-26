import torch
try:
    from cc_torch import connected_components_labeling
except ImportError:
    connected_components_labeling = None
from scipy.ndimage import label
from torch import Tensor


def connected_components(input: Tensor) -> Tensor:
    """Performs connected components. If tensor is on the GPU CC is performed on the GPU if not Scipy (CPU) is used.

    Args:
        input (Tensor): Tensor of the shape [H, W].

    Returns:
        output (Tensor): Connected components as a long tensor of the shape [H, W].
    """
    # Perform connected components.
    # cc_torch is CUDA-only; for CPU, MPS, or any non-CUDA accelerator fall back to scipy.
    if connected_components_labeling is None or not input.is_cuda:
        np_in = input.detach().cpu().numpy()
        output: Tensor = torch.from_numpy(label(np_in)[0]).long().to(input.device)
    else:
        output = connected_components_labeling(input.byte()).long()
        for index, value in enumerate(output.unique(sorted=True)):
            output[output == value] = index
    return output
