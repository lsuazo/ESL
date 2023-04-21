import numpy as np


def iterate_x_y_version(X, y, betas, active_mask):
    current_pred = X @ betas
    current_residual = y - current_pred
    current_corrs = X.T @ current_residual
    if sum(active_mask) == 0:
        active_idx = np.argmax(np.abs(current_corrs))
        active_mask[active_idx] = True

    # construct equi-angular vector
    signed_X = X.copy()
    signed_X[:, active_mask] *= np.sign(current_corrs[active_mask])
    onesA = np.ones(active_mask.sum())
    XA = signed_X[:, active_mask]
    GA = XA.T @ XA
    GA_inv = np.linalg.inv(GA)
    mixture_coeffs = GA_inv @ onesA
    vA = XA @ mixture_coeffs

    # find how much to move in that direction
    if sum(active_mask) < len(betas):
        temp_active_idx = np.argmax(active_mask)
        max_id_corr = np.abs(current_corrs[temp_active_idx])
        a = X.T @ vA
        left = (max_id_corr - current_corrs) / (1 - a)
        right = (max_id_corr + current_corrs) / (1 + a)
        left[active_mask] = np.inf
        right[active_mask] = np.inf
        left[left < 0] = np.inf
        right[right < 0] = np.inf
        min_left_right = np.minimum(left, right)
        new_active_idx = np.argmin(min_left_right)
        gamma = min_left_right[new_active_idx]
    else:
        gamma = current_residual.dot(vA) / (vA.dot(vA))
        new_active_idx = -1

    # convert to additions to betas
    delta_betas = mixture_coeffs * gamma
    iter_deltas = iter(delta_betas)
    expanded_deltas = np.zeros_like(betas)
    for i, flag in enumerate(active_mask):
        if flag:
            next_delta = next(iter_deltas)
            expanded_deltas[i] = next_delta * np.sign(current_corrs[i])

    return gamma, expanded_deltas, new_active_idx


def iterate_xtx_version(xtx, xty, betas, active_mask):
    ids = np.arange(0, len(betas))
    current_corrs = xty - xtx @ betas
    if sum(active_mask) == 0:
        active_idx = np.argmax(np.abs(current_corrs))
        active_mask[active_idx] = True

    signFlip = np.diag([np.sign(current_corrs[idx]) for idx in ids[active_mask]])
    sigmaA = signFlip @ xtx[ids[active_mask], :][:,
                        ids[active_mask]] @ signFlip  # weird way to index, but this keeps the desired shape
    sigmaA_inv = np.linalg.inv(sigmaA)
    mixing_coeffs = signFlip @ sigmaA_inv @ np.ones(sum(active_mask))
    xtxequi = xtx[:, ids[active_mask]] @ mixing_coeffs

    temp_active_idx = np.argmax(active_mask)
    max_id_corr = np.abs(current_corrs[temp_active_idx])

    if sum(active_mask) < len(betas):
        left = (max_id_corr - current_corrs) / (1 - xtxequi)
        right = (max_id_corr + current_corrs) / (1 + xtxequi)
        left[active_mask] = np.inf
        right[active_mask] = np.inf
        left[left < 0] = np.inf
        right[right < 0] = np.inf
        min_left_right = np.minimum(left, right)
        next_idx = np.argmin(min_left_right)
        gamma = min_left_right[next_idx]
    else:
        projected_residuals = mixing_coeffs.T @ (xty - xtx @ betas)
        gamma = projected_residuals / (mixing_coeffs.T @ xtx @ mixing_coeffs)
        next_idx = -1

    delta_betas = np.zeros_like(betas)
    delta_betas[active_mask] = gamma
    delta_betas[active_mask] *= mixing_coeffs

    return gamma, delta_betas, next_idx