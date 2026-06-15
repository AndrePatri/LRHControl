import numpy as np
import torch


def create_dataset(hf, name, data, **kwargs):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    arr = np.asarray(data)
    dataset_kwargs = {}
    if arr.shape != () and arr.dtype.kind not in ("O", "S", "U"):
        dataset_kwargs.update({
            "compression": "lzf",
            "shuffle": True,
            "chunks": True,
        })
    dataset_kwargs.update(kwargs)
    return hf.create_dataset(name, data=data, **dataset_kwargs)
