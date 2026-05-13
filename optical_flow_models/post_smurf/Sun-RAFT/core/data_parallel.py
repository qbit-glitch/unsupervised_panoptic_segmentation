from torch.nn.parallel._functions import Scatter
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch

def split_list(obj, target_gpus):
    assert isinstance(obj, list)
    assert len(obj) > 0
    size = len(obj) // len(target_gpus)
    return [obj[i * size:(i + 1) * size] for i in range(len(target_gpus))]

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            # return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            mapped = []
            # treat lists in dicts with key "full_img1" and "full_img2" differently. All other lists and dicts are treated like before. Previously all components of the lists were shared on all GPUs, but were sliced over channels within each element: splitting channels of images, instead of splitting the list (whole images).
            for k, v in obj.items():
                if k in ["full_img1", "full_img2"]:
                    v = split_list(v, target_gpus)
                else:
                    v = scatter_map(v)
                k = scatter_map(k)
                mapped.append(list(zip(k, v)))
            return list(map(type(obj), zip(*mapped)))

        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class ListDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
