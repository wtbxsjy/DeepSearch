import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DeepSearch.utils.probability import *
from pyteomics import pepxml, mzid
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

def standard_gaussian(x):
    return np.exp(-0.5 * np.sum(x**2, axis=-1)) / (2 * np.pi)**(x.shape[-1]/2)


def estimate_multivariate_density(data, xmin, xmax, ymin, ymax, zmin, zmax, hinv):
    bandWidthX = 1.0 / hinv[0, 0]
    bandWidthY = 1.0 / hinv[1, 1]
    bandWidthZ = 1.0 / hinv[2, 2]

    xvals = np.arange(xmin, xmax + bandWidthX, bandWidthX)
    yvals = np.arange(ymin, ymax + bandWidthY, bandWidthY)
    zvals = np.arange(zmin, zmax + bandWidthZ, bandWidthZ)

    X, Y, Z = np.meshgrid(xvals, yvals, zvals, indexing='ij')
    # Shape: (num_grid_points, 3)
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Calculate the difference between each grid point and each data point
    # Shape: (num_grid_points, num_data_points, 3)
    diff = grid_points[:, None, :] - data[None, :, :]

    # Apply the bandwidth matrix
    # Shape: (num_grid_points, num_data_points, 3)
    b = np.einsum('ijk,kl->ijl', diff, hinv)

    # Calculate the density at each grid point
    # Shape: (num_grid_points, num_data_points)
    densities = standard_gaussian(b)
    density_values = np.sum(densities, axis=1) * \
        np.linalg.det(hinv) / data.shape[0]

    density_values = density_values.reshape(X.shape)

    return xvals, yvals, zvals, density_values


def estimate_bivariate_density(data, xmin, xmax, ymin, ymax, hinv):
    bandWidthX = 1.0 / hinv[0, 0]
    bandWidthY = 1.0 / hinv[1, 1]

    xvals = np.arange(xmin, xmax + bandWidthX, bandWidthX, dtype=np.float32)
    yvals = np.arange(ymin, ymax + bandWidthY, bandWidthY, dtype=np.float32)

    X, Y = np.meshgrid(xvals, yvals)
    # Shape: (num_grid_points, 2)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1, dtype=np.float32)

    # Calculate the difference between each grid point and each data point
    # Shape: (num_grid_points, num_data_points, 2)
    diff = grid_points[:, None, :] - data[None, :, :]

    # Apply the bandwidth matrix
    # Shape: (num_grid_points, num_data_points, 2)
    b = np.einsum('ijk,kl->ijl', diff, hinv)
    del diff 
    # Calculate the density at each grid point
    # Shape: (num_grid_points, num_data_points)
    densities = standard_gaussian(b)
    del b
    zvals = np.sum(densities, axis=1) * np.linalg.det(hinv) / data.shape[0]

    zvals = zvals.reshape(X.shape)

    return xvals, yvals, zvals


def cal_pep2d(score_true, score_false, len_true, len_false):
    len_true = np.log(len_true + 1)
    len_false = np.log(len_false + 1)
    n_true = len(len_true)
    n_false = len(len_false)

    max_len = max(max(len_true), max(len_false))
    max_score = max(max(score_true), max(score_false))
    min_len = min(min(len_true), min(len_false))
    min_score = min(min(score_true), min(score_false))

    tt = max_len - min_len
    max_len += 0.1 * tt
    min_len -= 0.1 * tt
    tt = max_score - min_score
    max_score += 0.1 * tt
    min_score -= 0.1 * tt

    factor = np.power(n_true, 1/6)

    cov = np.cov(np.stack([score_true, len_true], axis=-1), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eig_transformed = np.array([factor / np.sqrt(di)
                                for di in eigenvalues])
    eig_transformed = np.diag(eig_transformed)
    hinv = eigenvectors @ eig_transformed @ eigenvectors.T
    x, y, true_z = estimate_bivariate_density(
        np.stack([score_true, len_true], axis=-1),  min_score, max_score, min_len, max_len, hinv)
    _, _, false_z = estimate_bivariate_density(np.stack(
        [score_false, len_false], axis=-1),  min_score, max_score, min_len, max_len, hinv)

    true_z[true_z == 0] = np.finfo(float).eps
    pep = np.where(false_z <= 0, np.finfo(float).eps, np.maximum(
        (false_z * 0.5) / true_z, np.finfo(float).eps))

    y_length = pep.shape[0]

    for j in range(y_length):
        # Find the index of the maximum value in the column
        maxInd = np.argmax(pep[j, :])
        maxVal = pep[j, maxInd]
        # Update all elements up to the maximum index with the maximum value
        if maxInd > 0:  # Check to ensure we don't update if maxInd is at the start
            pep[j, :maxInd] = maxVal

    return x, y, pep


def cal_pep3d(score_true, score_false, len_true, len_false, n_mod_true, n_mod_false):
    # Transform lengths and n_mods
    len_true = np.log(len_true + 1)
    len_false = np.log(len_false + 1)
    n_mod_true = np.log(n_mod_true + 1)
    n_mod_false = np.log(n_mod_false + 1)

    # Combine the three dimensions into a single dataset
    data_true = np.stack([score_true, len_true, n_mod_true], axis=-1)
    data_false = np.stack([score_false, len_false, n_mod_false], axis=-1)

    # Define the range for each dimension
    min_score, max_score = min(np.min(score_true), np.min(
        score_false)), max(np.max(score_true), np.max(score_false))
    min_len, max_len = min(np.min(len_true), np.min(len_false)), max(
        np.max(len_true), np.max(len_false))
    min_n_mod, max_n_mod = min(np.min(n_mod_true), np.min(
        n_mod_false)), max(np.max(n_mod_true), np.max(n_mod_false))

    tt = max_len - min_len
    max_len += 0.1 * tt
    min_len -= 0.1 * tt
    tt = max_score - min_score
    max_score += 0.1 * tt
    min_score -= 0.1 * tt
    tt = max_n_mod - min_n_mod
    max_n_mod += 0.1 * tt
    min_n_mod -= 0.1 * tt

    # Calculate the covariance and its inverse for the bandwidth matrix
    factor = np.power(len(len_true), 1/6)
    cov = np.cov(data_true, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eig_transformed = np.array([factor / np.sqrt(di)
                                for di in eigenvalues])
    eig_transformed = np.diag(eig_transformed)
    hinv = eigenvectors @ eig_transformed @ eigenvectors.T

    # Estimate the density for true and false data sets
    x, y, z, true_density = estimate_multivariate_density(
        data_true, min_score, max_score, min_len, max_len, min_n_mod, max_n_mod, hinv)
    _, _, _, false_density = estimate_multivariate_density(
        data_false, min_score, max_score, min_len, max_len, min_n_mod, max_n_mod, hinv)

    # Compute PEP
    true_z[true_z == 0] = np.finfo(float).eps
    pep = np.where(false_density <= 0, np.finfo(
        float).eps, (false_density * 0.5) / np.maximum(true_density, np.finfo(float).eps))

    for k in range(pep.shape[0]):
        for j in range(pep.shape[1]):
            # Find the index of the maximum value in the column
            maxInd = np.argmax(pep[k, j, :])
            maxVal = pep[k, j, maxInd]
            # Update all elements up to the maximum index with the maximum value
            if maxInd > 0:  # Check to ensure we don't update if maxInd is at the start
                pep[k, j, :maxInd] = maxVal

    return x, y, z, pep


def cal_pep(score_true, score_false, len_true, len_false):
    len_true = np.log(len_true + 1)
    len_false = np.log(len_false + 1)
    n_true = len(len_true)
    n_false = len(len_false)

    max_len = max(max(len_true), max(len_false))
    max_score = max(max(score_true), max(score_false))
    min_len = min(min(len_true), min(len_false))
    min_score = min(min(score_true), min(score_false))

    tt = max_len - min_len
    max_len += 0.1 * tt
    min_len -= 0.1 * tt
    tt = max_score - min_score
    max_score += 0.1 * tt
    min_score -= 0.1 * tt

    factor = np.power(n_true, 1/6)

    cov = np.cov(np.stack([score_true, len_true], axis=-1, dtype=np.float32), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eig_transformed = np.array([factor / np.sqrt(di)
                                for di in eigenvalues])
    eig_transformed = np.diag(eig_transformed)
    hinv = np.float32(eigenvectors @ eig_transformed @ eigenvectors.T)
    x, y, true_z = estimate_bivariate_density(
        np.stack([score_true, len_true], axis=-1, dtype=np.float32),  min_score, max_score, min_len, max_len, hinv)
    _, _, false_z = estimate_bivariate_density(np.stack(
        [score_false, len_false], axis=-1, dtype=np.float32),  min_score, max_score, min_len, max_len, hinv)

    true_z[true_z == 0] = np.finfo(float).eps
    pep = np.where(false_z <= 0, np.finfo(float).eps, np.maximum(
        (false_z * 0.5) / true_z, np.finfo(float).eps))

    y_length = pep.shape[0]

    for j in range(y_length):
        # Find the index of the maximum value in the column
        maxInd = np.argmax(pep[j, :])
        maxVal = pep[j, maxInd]
        # Update all elements up to the maximum index with the maximum value
        if maxInd > 0:  # Check to ensure we don't update if maxInd is at the start
            pep[j, :maxInd] = maxVal

    return x, y, pep



def control_FDR(hits: pd.DataFrame, FDR: float, key: str = 'score', use_pep: bool = False):
    hits = hits.sort_values(by=key, ascending=(
        not use_pep)).reset_index(drop=True)
    total_hits = len(hits)
    total_peptide_hit = hits['decoy'].value_counts()['-']
    total_decoy_hit = hits['decoy'].value_counts()['+']
    assert total_hits == total_peptide_hit + total_decoy_hit

    curr_peptide_hit = total_peptide_hit
    curr_decoy_hit = total_decoy_hit
    fdr_arr = [0.] * total_hits
    q_arr = [0.] * total_hits
    for i, row in hits.iterrows():
        if curr_peptide_hit == 0 or curr_decoy_hit == 0:
            break
        fdr_i = (curr_decoy_hit + 1) / curr_peptide_hit
        if row['decoy'] == '-':
            curr_peptide_hit -= 1
        else:
            curr_decoy_hit -= 1

        fdr_arr[i] = fdr_i

    hits['fdr'] = fdr_arr

    min_fdr = 10000.
    control_position = 0
    for i, fdr in enumerate(fdr_arr):
        min_fdr = min(min_fdr, fdr)
        q_arr[i] = min_fdr
        if q_arr[i] >= FDR:
            control_position = i
    hits['q_value'] = q_arr
    # return hits, control_position
    controled_hits = hits.iloc[control_position:]
    return controled_hits, q_arr


def calculate_pep(hits: pd.DataFrame):
    xgrid, ygrid, peps = cal_pep(hits[hits['decoy'] == '-']['score'], hits[hits['decoy'] == '+']
                                 ['score'], hits[hits['decoy'] == '-']['seq_len'], hits[hits['decoy'] == '+']['seq_len'])
    f = RegularGridInterpolator([xgrid, ygrid], peps.T)
    data = (hits['score'].to_numpy(), np.log(hits['seq_len'].to_numpy()))

    return f(data, method='linear')


def main():
    result_dir = Path('/mnt/data1/NM/run/xy/HEK293/tryptic2/')

    hits = pd.read_csv(
        result_dir/'all_PSM.tsv', sep='\t')
    hits = hits[hits['score'].isna() != True]
    hits = hits[hits['charge'] <= 4]
    hits['seq_len'] = hits['peptide'].str.len()
    hits['n_mod'] = hits['modified_peptide'].str.count('\(')
    use_var_mods = hits['n_mod'].max() > 0
    n_spectra = len(hits)

    pep = calculate_pep(hits)
    hits['pep'] = pep
    controled_hits, q_arr = control_FDR(hits, 0.01, key='pep', use_pep=True)


if __name__ == '__main__':
    main()
