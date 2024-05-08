import importlib
import torch
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """extract: extract index given time step t for a batch of data

    Args:
        a (torch.Tensor): _description_
        t (torch.Tensor): _description_
        x_shape (torch.Size): _description_

    Returns:
        torch.Tensor: _description_
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def modulo_with_wrapped_range(
    vals, range_min: float = -1., range_max: float = 1.
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    return retval


# print(modulo_with_wrapped_range(1, -2, 2))


def estimate_tensor_size( 
                        dtype: torch.dtype = torch.float32,
                        shape: torch.Size = (1, 1, 1)):
    """estimate_chunk_size: estimate the chunk size for tensor given the max memory
                            shape: batch first
                            max_mem: in bytes
    """
    placeholder = 1
    for dim in shape[1:]:
        placeholder *= dim

    if dtype == torch.float32:
        byte = 4
    elif dtype == torch.bfloat16:
        byte = 2
    else:
        raise NotImplementedError('Only support float32 and bfloat16')

    tensor_size = placeholder * byte
    return tensor_size



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_ckpt():

    return


