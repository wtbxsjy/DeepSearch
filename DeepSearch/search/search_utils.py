import pickle
import DeepSearch.utils.peptide as pp
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import torch
import numpy as np
import h5py as h5

from line_profiler import profile


def save_bucket(path: Path, bucket: Dict, peptide2idx: Dict, bucket_idx_range:Dict, peptide_meta: Dict):
    with open(path, 'wb') as f:
        pickle.dump(bucket, f)
        pickle.dump(peptide2idx, f)
        pickle.dump(bucket_idx_range, f)
        pickle.dump(peptide_meta, f)


def load_bucket(path: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    with open(path, 'rb') as f:
        bucket = pickle.load(f)
        peptide2idx = pickle.load(f)
        bucket_idx_range = pickle.load(f)
        peptide_meta = pickle.load(f)
    return bucket, peptide2idx, bucket_idx_range, peptide_meta


# def cal_pair_bias(max_charge, pep,
#                  context_len: int=32,
#                  bias_max_charge: int=6,
#                  mods: Optional[Tuple[Tuple]]=None) -> np.ndarray:
#        assert bias_max_charge >= max_charge, f"bias_max_charge {bias_max_charge} should be larger than max_charge {max_charge}"
#
#        neutral_mass = pp.cal_pep_mass(pep, mods)
#        #neutral_mass = pre_mz * max_charge - max_charge * pp.PROTON
#        pair_bias = []
#        pep_len = len(pep)
#        padding_len = context_len - pep_len
#
#        for charge in range(1, bias_max_charge + 1):
#            for ion in 'aby':
#                for ion_mod in ('NH3', 'H2O'):
#                    if charge <= max_charge:
#                        t = pp.cal_theoretical_fragmentation(pep,
#                                                                          charge=charge,
#                                                                          ion_type=ion,
#                                                                          ion_mod=ion_mod,
#                                                                          mods=mods,
#                                                                          pre_neutral_mass=neutral_mass,
#                                                                          append_proton=True, append_pep=True)
#                        frags = np.array(t, dtype=np.float32)
#
#                        pair_bias.append(np.subtract.outer(frags, frags))
#                    else:
#                        pair_bias.append(np.zeros((pep_len+2, pep_len+2)))
#
#        pair_bias = np.stack(pair_bias, -1, dtype=np.float32)
#        pair_bias = np.pad(pair_bias, ((0, padding_len), (0, padding_len), (0, 0)),
#                           mode="constant", constant_values=0)
#        return pair_bias

def cal_pair_bias(max_charge, pep,
                  context_len: int = 32,
                  bias_max_charge: int = 6,
                  mods: Optional[Tuple[Tuple]] = None,
                  neutral_mass: Optional[float] = None) -> np.ndarray:
    assert bias_max_charge >= max_charge, f"bias_max_charge {bias_max_charge} should be larger than max_charge {max_charge}"
    if neutral_mass is None:
        neutral_mass = pp.cal_pep_mass(pep, mods) + pp.H2O
    pair_bias = np.zeros((context_len+2, context_len+2,
                         6 * bias_max_charge), dtype=np.float32)
    pep_len = len(pep)
    i = 0
    mask = np.ones((context_len+2, context_len+2), dtype=bool)
    mask[pep_len+2:, :] = False
    mask[:, pep_len+2:] = False
    for charge in range(1, bias_max_charge + 1):
        for ion in 'aby':
            for ion_mod in ('NH3', 'H2O'):
                if charge <= max_charge:
                    t = pp.cal_theoretical_fragmentation(pep,
                                                         [0] *
                                                         (context_len + 2),
                                                         charge=charge,
                                                         ion_type=ion,
                                                         ion_mod=ion_mod,
                                                         mods=mods,
                                                         pre_neutral_mass=neutral_mass,
                                                         append_proton=True, append_pep=True)

                    np.subtract.outer(t, t, out=pair_bias[:, :, i], where=mask)
                i += 1

    return pair_bias


def cal_ppm_error(t_mz: float, e_mz: float):
    ppm = (-t_mz + e_mz) / t_mz * 10**6
    return ppm


def cal_precursor_ppm_error(pre_mz: float, pep_mass: float, charge: int):
    """cal_precursor_ppm_error calculate precursor ppm error give peptide and spectrum precursor mz

    Args:
        pre_mz (float): _description_
        peptide (str): _description_
        charge (int): _description_

    Returns:
        _type_: _description_
    """
    theoratical_mz = (pep_mass + pp.H2O + pp.PROTON * charge) / charge
    ppm_error = cal_ppm_error(theoratical_mz, pre_mz)
    return ppm_error


def extract_peptide_meta(pep_meta: Dict, pep2idx: Dict, peptide: str,) -> Dict:
    idx = pep2idx[peptide]
    pep_mass = pep_meta['pep_mass'][idx]
    ori_pep = pep_meta['peptide'][idx]
    dec = pep_meta['dec'][idx]
    return {
        'neutral_mass': pep_mass,
        'peptide': ori_pep,
        'protein': dec,
        'mods': pep_meta['mods'][idx],
    }


def retrieve_candidates_emds_nocharge(all_emds: np.ndarray, pep2idx: Dict, peptides: List[str]) -> torch.Tensor:
    indices = []
    for pep in peptides:
        idx = pep2idx[pep]
        indices.append(idx)
    emds = all_emds[indices, :]
    return torch.from_numpy(emds)

def retrieve_candidates_close(bucket_idx_range: Dict, peptide_bucket: Dict, charge, mass, pre_mz, precursor_ppm=10, delta = 1) -> torch.Tensor:
    peptide_candidates = []
    peptide_ppms = []
    indices = []
    for delta_ in range(-delta, delta+1):
        rounded_mass = str(round(mass + delta_ * 0.1, 1))
        if rounded_mass in peptide_bucket:
            idx = bucket_idx_range[rounded_mass]
            peptides = peptide_bucket[rounded_mass]
            for i in idx:
                peptide, pep_mass = peptides[i - idx[0]]
                ppm = cal_precursor_ppm_error(pre_mz, pep_mass, charge)
                if abs(ppm) < precursor_ppm:
                    peptide_candidates.append(peptides[i - idx[0]])
                    peptide_ppms.append(ppm)
                    indices.append(i)
    return peptide_candidates, peptide_ppms, np.array(indices, dtype=np.int64)

@profile
def retrieve_candidates_open(peptide_bucket: Dict, bucket_idx_range: Dict, mass: float, delta=3500, resolution=0.1) -> Tuple[List[str], List[Tuple]]:
    peptide_candidates = []
    indices = []
    for delta_ in range(-1500, delta+1):
        rounded_mass = str(round(mass + delta_ * resolution, 1))
        if rounded_mass in peptide_bucket:
            #t, _ = zip(*peptide_bucket[rounded_mass])
            peptide_candidates.extend(peptide_bucket[rounded_mass])
            indices.extend(bucket_idx_range[rounded_mass])
    return peptide_candidates, np.array(indices, dtype=np.int64)


def retrieve_peptide_candidate(peptide_bucket: Dict, mass: float, delta=1, resolution=0.1) -> List[str]:
    peptide_candidates = []
    for delta in range(-delta, delta+1):
        rounded_mass = str(round(mass + delta * resolution, 1))
        if rounded_mass in peptide_bucket:
            peptide_candidates.extend(peptide_bucket[rounded_mass])
    return peptide_candidates
