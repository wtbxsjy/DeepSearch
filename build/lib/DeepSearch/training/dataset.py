from typing import Dict, Iterator, Literal
import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, IterableDataset
import logging
from typing import *
from DeepSearch.utils.spectrum import binning_spectrum_tensor
import DeepSearch.utils.peptide as pp 
import re
import h5py as h5
from pyteomics import mgf
import itertools

# canonical spectrum-peptide pair dataset


class Normalize(object):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.reshape((2, 1))
        self.std = std.reshape((2, 1))
        return 
    
    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        norm_spectrum = (spectrum - self.mean) / self.std 
        return norm_spectrum


class CanonicalPSMDataset(Dataset):
    """CanonicalPSMDataset _summary_

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    pep_feature_names = {
        "ion_types": [
            'b',
            'b-H2O',
            'b-NH3',
            'y',
            'y-H2O',
            'y-NH3',
            'a'
        ],
        "max_charge": 10,

        "features": [
            'theoretical_mz',
            'experimental_mz',
            'experimental_intensity',
            'seq_aa_mass',
            'charge',
            'pre_mass',
        ]
    }

    def __init__(self,
                 mshdf5: str,
                 config: Dict,
                 norm: bool = True,
                 augmentation_prob = 0.0,
                 masking_prob = 0.1,
                 max_charge = 6
                 ) -> None:
        """__init__ _summary_

        Args:
            data_file (str): _description_
            split (Literal["train", "test", "validation"]): _description_. Defaults to None.
            max_pep_length (int, optional): _description_. Defaults to 32.
            min_pep_length (int, optional): _description_. Defaults to 6.
            max_n_peaks (int, optional): _description_. Defaults to 512.
            max_mz (float, optional): _description_. Defaults to 3500..
            wrap_mz (bool, optional): _description_. Defaults to True.
            wrap_intensity (bool, optional): _description_. Defaults to True.
            n_thread (int, optional): _description_. Defaults to 16.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.config = config 
        self.max_n_peaks = config['max_npeaks']
        self.max_mz = config['max_mz']
        self.max_pep_len = config['max_pep_len']
        self.start_mz = config['start_mz']
        self.norm = norm
        self.max_charge = max_charge

        self.h5f = h5.File(mshdf5, 'r')

        self.n_items = self.h5f['spectra'].shape[0]
        self.masking_prob = masking_prob
        self.augmentation_prob = augmentation_prob


    def __featurize_spectrum(self, spectrum):
        mz_array = spectrum[0]
        intensities = spectrum[1]
        n_peaks = len(intensities)

        # rank
        temp = intensities.argsort()
        ranks = np.empty_like(mz_array)
        ranks[temp] = np.arange(n_peaks) / n_peaks

        def get_window_index(i, delta_mz=50):
            start_idx, end_idx = 0, 0
            mz = mz_array[i]
            for j in range(i, -1, -1):
                if mz_array[j] >= mz - delta_mz:
                    start_idx = j
                else:
                    break
            for j in range(i, n_peaks):
                if mz_array[j] <= mz + delta_mz:
                    end_idx = j + 1
                else:
                    break

            assert end_idx > start_idx
            return start_idx, end_idx

        # local info
        local_rank = np.empty_like(mz_array)
        local_significance = np.empty_like(mz_array)
        local_relative_int = np.empty_like(mz_array)

        for i in range(n_peaks):
            _ , intensity = mz_array[i], intensities[i]
            start_idx, end_idx = get_window_index(i, 50)
            local_intensities = intensities[start_idx: end_idx]
            n_local_peaks = len(local_intensities)
            assert n_local_peaks > 0

            local_significance[i] = np.tanh(
                intensity/(local_intensities.min() + 1e-7) - 1)
            local_relative_int[i] = intensity / (local_intensities.max() + 1e-7)
            temp = local_intensities.argsort().argsort()
            local_rank[i] = temp[i - start_idx] / n_local_peaks

        return np.array([mz_array, intensities, ranks, local_relative_int, local_significance, local_rank])

    def __pad_spectrum(self, spectrum, pad_len):
        n_peaks = spectrum.shape[1]
        spectrum = np.pad(spectrum, ((0, 0), (0, pad_len - n_peaks)),
                              mode="constant", constant_values=0)
        return spectrum
    
    def __get_spectrum(self, spectrum) -> np.ndarray:
        mz_array = spectrum[0]
        intensities = spectrum[1]

        # drop peaks out of mz ranges
        idx = mz_array < self.max_mz
        _mz_array = np.array(mz_array)[idx]
        _intensities = np.array(intensities)[idx] 
        _intensities /= _intensities.max()

        n_peaks = len(_mz_array)

        assert len(_intensities) == n_peaks

        if self.augmentation_prob > 0:
            rng = np.random.default_rng()
            if rng.random() < self.augmentation_prob:
                n_peaks_to_keep = int(n_peaks * self.masking_prob)
                n_peaks_to_keep = rng.integers(n_peaks_to_keep, n_peaks)
                index = rng.choice(n_peaks, n_peaks_to_keep, replace=False)
                if n_peaks_to_keep >= 32:
                    _mz_array[index] = 0.
                    _intensities[index] = 0.
            

        # select peaks if exceeds max_n_peaks based on intensity
        if n_peaks > self.max_n_peaks:
            indexes = sorted(range(0, n_peaks),
                             key=lambda x: intensities[x])[-self.max_n_peaks:]
            indexes = sorted(indexes)

            _mz_array = _mz_array[indexes]
            _intensities = _intensities[indexes]

            assert len(_mz_array) == self.max_n_peaks
        if self.norm:
            _intensities = np.sqrt(_intensities)
        return np.array([_mz_array, _intensities])


    def __parse_mod_seq(self, mod_seq: str) -> str:
        mods = list(filter(None, re.split('\(|\)|_', mod_seq)))
        seq = []
        u_seq = []
        aa_idx = 0
        for m in mods:
            if m == 'ox':
                assert seq[aa_idx - 1] == 'M'
                seq[aa_idx - 1] = 'm'
                u_seq[aa_idx - 1] = 'M'
            else:
                for aa in m:
                    mod_aa = aa
                    seq.append(mod_aa)
                    u_seq.append(aa)
                    aa_idx += 1
        return "".join(seq)


    def tokenize_peptides(self, mod_seq: str) -> torch.Tensor:
        # Tokenize peptide as in IUPAC Dict

        max_pep_len = self.config['max_pep_len']
        pep_len = len(mod_seq)
        t = np.zeros((2 + max_pep_len), np.int64) # [s] + pep_len + [e]
        t[0] = 1   # [s]

        for i in range(1, pep_len+1):
            t[i] = pp.AA_INDEX[mod_seq[i - 1]]
        t[pep_len + 1] = 24 # [e]
        
        return torch.tensor(t, dtype=torch.long)
    

    def __tensorize_charge(self, charge: int) -> torch.Tensor:
        # one hot encoding charge
        t = torch.zeros((self.config['max_charge']))
        t[charge - 1] = 1.
        # t = F.one_hot(torch.tensor(charge - 1), self.config['max_charge'])
        return t


    def __getitem__(self, index: int) -> Dict:
        """__getitem__ _summary_

        Args:
            index (int): _description_

        Returns:
            Dict: _description_
        """
        
        spectrum = self.h5f['spectra'][index].astype(np.float32)
        spectrum = np.reshape(spectrum, (2, -1))

        # use charge, mz from mgf
        charge = self.h5f['charges'][index][0].item()
        pre_mz = self.h5f['premzs'][index][0].astype(np.float32)
        #energy = (self.h5f['energies'][index][0] / 35.) .astype(np.float32)
        mod_seq = self.h5f['peptides'][index].decode('utf-8')

        mod_seq = self.__parse_mod_seq(mod_seq)
        spectrum = self.__get_spectrum(spectrum)
        npeaks = spectrum.shape[1]
        #spectrum_feature = self.__featurize_spectrum(spectrum)
        t_peptide = self.tokenize_peptides(mod_seq)
        t_charge = self.__tensorize_charge(charge)

        # pep_mask = torch.arange(self.max_pep_len + 1) >= len(mod_seq) + 1
        pep_mask = (t_peptide != 0).long()
        # assert charge == msms_row['Charge']
        
        ret = {"spectrum": spectrum[:],
               #"feature": spectrum_feature,
               "pre_mz": pre_mz,
               "charge": charge,
               "seq": mod_seq,
               #"energy": energy,
               "npeaks": npeaks,
               "t_peptide": t_peptide,
               "t_charge": t_charge,
               "pep_mask": pep_mask}

        return ret


    def __len__(self):
        return self.n_items


class BinnedSpectrumDataset(CanonicalPSMDataset):
    # dataset with spectrum binning

    def __init__(self, mshdf5: str, config: Dict, norm: bool = True, mz_step: float = 0.1, use_meta: bool = False) -> None:
        super().__init__(mshdf5, config, norm)
        self.mz_step = mz_step
        self.use_meta = use_meta

    def __getitem__(self, index: int) -> Dict:
        spectrum_info = super().__getitem__(index)
        spectrum = spectrum_info['spectrum']
        #feature = spectrum_info['feature']
        n_peaks = spectrum.shape[1]
        # assert n_peaks == feature.shape[1]
        # concat charge HCD for each peak
        #if self.use_meta:
            #charge = np.array([[spectrum_info['charge']] * n_peaks])
            #energy = np.array([[spectrum_info['energy']] * n_peaks])
            #energy = (energy - 15) / 25
            #feature = np.concatenate([feature, charge, energy], axis=0)

        # padding
        if n_peaks < self.max_n_peaks:
            spectrum = np.pad(spectrum, ((0, 0), (0, self.max_n_peaks - n_peaks)),
                              mode="constant", constant_values=0)
            #feature = np.pad(feature, ((0, 0), (0, self.max_n_peaks - n_peaks)),
            #                  mode="constant", constant_values=0)

        spectrum = torch.tensor(spectrum.transpose(), dtype=torch.float)
        #feature = torch.tensor(feature.transpose(), dtype=torch.float)
        binned = binning_spectrum_tensor(
            spectrum, self.start_mz, self.max_mz, self.mz_step)
        mask = ~(spectrum.sum(dim=1).bool())
        spectrum_info['mask'] = mask
        spectrum_info['binned'] = binned 
        spectrum_info['spectrum'] = spectrum
        #spectrum_info['feature'] = feature
        return spectrum_info

    def __len__(self):
        return super().__len__()


class SpectrumDataset(CanonicalPSMDataset):
    def __init__(self, 
                 mshdf5: str, 
                 config: Dict, 
                 norm: bool = True, 
                 augmentation_prob = 0.0, 
                 masking_prob = 0.1) -> None:
        super().__init__(mshdf5, config, norm, augmentation_prob, masking_prob)
        self.bias_max_charge = self.max_charge
        if 'bias_max_charge' in config:
            self.bias_max_charge = config['bias_max_charge']


    def cal_pair_bias(self, pre_mz, max_charge, pep):
        neutral_mass = pre_mz * max_charge - max_charge * pp.PROTON
        pair_bias = []
        pep_len = len(pep)
        padding_len = self.max_pep_len - pep_len
        #max_charge = 3 if max_charge > 3 else max_charge
        for charge in range(1, self.bias_max_charge + 1):
            for ion in 'aby':
                for ion_mod in ('NH3', 'H2O'):
                    if charge <= max_charge:
                        frags = np.array(pp.cal_theoretical_fragmentation(pep, 
                                                                          charge=charge, 
                                                                          ion_type=ion,
                                                                          ion_mod=ion_mod, 
                                                                          pre_neutral_mass=neutral_mass, 
                                                                          append_proton=True, append_pep=True), dtype=np.float32)

                        pair_bias.append(np.subtract.outer(frags, frags))
                    else:
                        pair_bias.append(np.zeros((pep_len+2, pep_len+2)))
        
        pair_bias = np.stack(pair_bias, -1)
        pair_bias = np.pad(pair_bias, ((0, padding_len), (0, padding_len), (0, 0)),
                           mode="constant", constant_values=0)
        return pair_bias


    def __getitem__(self, index: int) -> Dict:
        spectrum_info = super().__getitem__(index)
        spectrum = spectrum_info["spectrum"]
        #feature = spectrum_info['feature']

        n_peaks = spectrum.shape[1]
        #assert n_peaks == feature.shape[1]
        if n_peaks < self.max_n_peaks:
            spectrum = np.pad(spectrum, ((0, 0), (0, self.max_n_peaks - n_peaks)),
                              mode="constant", constant_values=0)
            
            #feature = np.pad(feature, ((0, 0), (0, self.max_n_peaks - n_peaks)),
            #                  mode="constant", constant_values=0)
            

        spectrum = torch.tensor(spectrum.transpose(), dtype=torch.float)
        #feature = torch.tensor(feature.transpose(), dtype=torch.float)
        mask = spectrum[:, 0] == 0

        spectrum_info['mask'] = mask
        spectrum_info['spectrum'] = spectrum
        
        #spectrum_info['feature'] = feature
        seq = spectrum_info['seq']
        pre_mz = spectrum_info['pre_mz']
        charge = spectrum_info['charge']
        pair_bias = self.cal_pair_bias(pre_mz, charge, seq)
        spectrum_info['pair_bias'] = torch.tensor(pair_bias, dtype=torch.float)
        spectrum_info['pep_mass'] = np.float32(pre_mz * charge - (charge - 1) * pp.PROTON)
        #print(type(spectrum_info['pep_mass']))
        return spectrum_info

    def __len__(self):
        return super().__len__()
    

class PairSpectrumDataset(CanonicalPSMDataset):
    def __init__(self, mshdf5: str, config: Dict, norm: bool = True, augmentation_prob=0, masking_prob=0.1) -> None:
        super().__init__(mshdf5, config, norm, augmentation_prob, masking_prob)
    
    #TODO: test other encoding for pairwise intensity
    def get_pair_representation(self, spectrum: np.ndarray):
        zero_pad_spectrum = np.pad(spectrum, ((0, 0), (1, 0) ))
        pair_rep = []
        mz = spectrum[0]
        intensity = spectrum[1]
        n_peaks = len(spectrum)
        pairwise_mz = np.subtract.outer(mz, mz)
        pair_rep.append(pairwise_mz)

        pairwise_int_r = np.tile(intensity[None,:], [n_peaks, 1])
        pairwise_int_c = np.tile(intensity[:, None], [1, n_peaks])
        pair_rep.extend([pairwise_int_r, pairwise_int_c])
        #TODO add charge and pre_mz feature
        return np.stack(pair_rep, -1)

         
    
    def __getitem__(self, index: int) -> Dict:
        spectrum_info = super().__getitem__(index)
        spectrum = spectrum_info["spectrum"]
        n_peaks = spectrum.shape[1]
        pair_rep = self.get_pair_representation(spectrum) # [N + 1, N + 1, D]

        if n_peaks < self.max_n_peaks:
            pad_len = self.max_n_peaks - n_peaks
            spectrum = np.pad(spectrum, ((0, 0), (0, pad_len)),
                              mode="constant", constant_values=0)
            pair_rep = np.pad(pair_rep, ((0, pad_len), (0, pad_len), (0, 0)))
        
        spectrum = torch.tensor(spectrum.transpose(), dtype=torch.float)
        pair_rep = torch.tensor(pair_rep, dtype=torch.float)
        mask = spectrum[:, 0] == 0





        spectrum_info['mask'] = mask
        spectrum_info['spectrum'] = spectrum
        spectrum_info['pair_rep'] = pair_rep
        # spectrum_info['feature'] = feature
        return spectrum_info



class NormalizedSpectrumDataset(CanonicalPSMDataset):
    def __init__(self, mshdf5: str, config: Dict, norm: bool = True, transform=Normalize) -> None:
        super().__init__(mshdf5, config, norm)
        self.transform = transform


    def __getitem__(self, index: int) -> Dict:
        spectrum_info = super().__getitem__(index)
        spectrum = spectrum_info["spectrum"]
        #feature = spectrum_info['feature']

        n_peaks = spectrum.shape[1]
        # assert n_peaks == feature.shape[1]
        if n_peaks < self.max_n_peaks:
            spectrum = np.pad(spectrum, ((0, 0), (0, self.max_n_peaks - n_peaks)),
                              mode="constant", constant_values=0)
            #feature = np.pad(feature, ((0, 0), (0, self.max_n_peaks - n_peaks)),
            #                 mode="constant", constant_values=0)

        spectrum = torch.tensor(spectrum.transpose(), dtype=torch.float)
        mask = spectrum[:, 0] == 0

        spectrum_info['mask'] = mask
        spectrum_info['spectrum'] = spectrum
        return spectrum_info

    def __len__(self):
        return super().__len__()


# wrap a mgf or multiple mgf file stream
class SpectrumOnlyDataset(IterableDataset):
    def __init__(self, mgf_name, max_mz=2000., norm=True, max_n_peaks=512) -> None:
        super().__init__()
        self.stream = mgf.MGF(mgf_name, convert_arrays=1, read_charges=False)
        self.max_mz = max_mz
        self.norm = norm
        self.max_n_peaks = max_n_peaks
    
    def __get_spectrum(self, mz_array, intensities) -> np.ndarray:

        # drop peaks out of mz ranges
        idx = mz_array < self.max_mz
        _mz_array = np.array(mz_array)[idx]
        _intensities = np.array(intensities)[idx]
        _intensities /= _intensities.max()

        n_peaks = len(_mz_array)

        assert len(_intensities) == n_peaks

        # select peaks if exceeds max_n_peaks based on intensity
        if n_peaks > self.max_n_peaks:
            indexes = sorted(range(0, n_peaks),
                             key=lambda x: intensities[x])[-self.max_n_peaks:]
            indexes = sorted(indexes)

            _mz_array = _mz_array[indexes]
            _intensities = _intensities[indexes]

            assert len(_mz_array) == self.max_n_peaks
        if self.norm:
            _intensities = np.sqrt(_intensities)
        return np.array([_mz_array, _intensities])

    
    def extract_spectrum(self, spectrum_meta):
        pre_mz = spectrum_meta['params']['pepmass'][0]
        charge = spectrum_meta['params']['charge'][0]
        rt = spectrum_meta['params']['rtinseconds']
        mz = spectrum_meta['m/z array']
        intensity = spectrum_meta['intensity array']
        # use relative intensity?
        spectrum = self.__get_spectrum(mz, intensity)
        n_peaks = spectrum.shape[1]
        # assert n_peaks == feature.shape[1]
        if n_peaks < self.max_n_peaks:
            spectrum = np.pad(spectrum, ((0, 0), (0, self.max_n_peaks - n_peaks)),
                              mode="constant", constant_values=0)
            # feature = np.pad(feature, ((0, 0), (0, self.max_n_peaks - n_peaks)),
            #                 mode="constant", constant_values=0)

        spectrum = torch.tensor(spectrum.transpose(), dtype=torch.float)
        # feature = torch.tensor(feature.transpose(), dtype=torch.float)
        mask = spectrum[:, 0] == 0

        return {
            'mask': mask,
            'spectrum': spectrum,
            'charge': charge,
            'pre_mz': pre_mz
        }


    def __iter__(self) -> Iterator:
        self.stream.reset()
        spectrum_iter = map(self.extract_spectrum, self.stream)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return spectrum_iter
        else:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
            spectrum_iter = itertools.islice(
                spectrum_iter, worker_id, None, worker_total_num)
            return spectrum_iter


def main():
    from torch.utils.data import DataLoader
    import yaml
    f = open('config/ddpm_bert_base.yaml')
    config = yaml.safe_load(f)['Meta']
    
    logging.basicConfig(level="INFO")
    # mn = np.array([535.251784546975, 0.03971371638870874])
    # std = np.array([306.1836847540895, 0.1076798530215255])
    
    dset = SpectrumDataset('data/test_train.h5', config)
    #dset = SpectrumOnlyDataset(
    #    mgf_name='/mnt/storage/human_synthetic/HCD_ALL/mgf/01640c_BH3-Thermo_SRM_Pool_24_01_01-3xHCD-1h-R2.mgf')
    loader = DataLoader(dset, 32,
                        num_workers=16, persistent_workers=True)
    
    it = loader.__iter__()

    x = next(it)
    next(it)
    # plot_spectrum(d)
    f.close()


if __name__ == "__main__":
    main()
