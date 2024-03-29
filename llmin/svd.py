import torch

def _low_rank_decomposition(weight, reduced_rank=64):
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param reduced_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    if weight.dim() != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {weight.dim()} dimensions.")

    # SVD Decomposition
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Truncated matrices
    A = Vh[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A, B

def decompose_delta_weight(new_weight, base_weight, alpha, reduced_rank, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    new_weight = new_weight.to(device)
    base_weight = base_weight.to(device)

    """
    Decompose the delta weight into low-rank matrices A and B, considering the alpha scaling factor.

    :param new_weight: The updated weight matrix after applying LoRA.
    :param base_weight: The original weight matrix before LoRA.
    :param alpha: The alpha scaling factor used in LoRA.
    :param reduced_rank: The rank for the low-rank decomposition.
    :return: A tuple of tensors (A, B)
    """
    delta_weight = new_weight - base_weight

    # Check if alpha is applied uniformly
    # Adjust the implementation if alpha is applied differently
    adjusted_delta_weight = delta_weight / alpha

    A, B = _low_rank_decomposition(adjusted_delta_weight, reduced_rank=reduced_rank)

    return A, B