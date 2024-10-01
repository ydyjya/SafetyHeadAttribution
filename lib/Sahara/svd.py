import torch


def svd_decomposition(tensor):
    U, S, V = torch.svd(tensor)
    return U, S, V


def normalize_columns(u):
    norms = torch.norm(u, dim=0)
    U_normalized = u / norms
    return U_normalized


def compute_subspace_similarity(matrix1, matrix2):
    u1, _, v1 = torch.linalg.svd(matrix1, full_matrices=False)
    u2, _, v2 = torch.linalg.svd(matrix2, full_matrices=False)
    S = torch.matmul(u1[:, :1].T, u2[:, :1])
    _, singular_values, _ = torch.linalg.svd(S)
    principal_angles = torch.acos(torch.clamp(singular_values, -1, 1))
    principal_angles_degrees = principal_angles * 180 / torch.pi
    return principal_angles_degrees.item()


def compute_subspace_spectral_norm(u1, u2):
    u1_normalized = normalize_columns(u1)
    u2_normalized = normalize_columns(u2)
    S = torch.matmul(u1_normalized[:, :1].T, u2_normalized[:, :1])

    _u, singular_values, _v = torch.linalg.svd(S)
    principal_angles = torch.acos(torch.clamp(singular_values, -1, 1))
    principal_angles_degrees = principal_angles * 180 / torch.pi
    return principal_angles_degrees.item()
