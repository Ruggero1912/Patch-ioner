import torch

def get_pseudo_inverse(A):
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Compute the pseudo-inverse of the singular values
    S_pinv = torch.zeros_like(S)
    non_zero = S > 1e-10  # Tolerance for considering a singular value as zero
    S_pinv[non_zero] = 1.0 / S[non_zero]

    # Construct the pseudo-inverse
    A_pinv = Vh.T @ torch.diag(S_pinv) @ U.T

    return A_pinv

def revert_transformation(features, linear_layer=None, A_pinv=None, b=None):
    assert linear_layer is not None or (A_pinv is not None and b is not None), "revert_transformation needs either the pseudo inverse od the linear layer to calculate the pseudo inverse from"
    if A_pinv is None:
        W = linear_layer.weight
        b = linear_layer.bias
        
        A_pinv = get_pseudo_inverse(W)
    
    return (features - b) @ A_pinv.t()