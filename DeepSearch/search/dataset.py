import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np 

import DeepSearch.utils.peptide as pp
from DeepSearch.utils.tokenizer import Tokenizer
from DeepSearch.search.search_utils import cal_pair_bias

import Bio.SeqIO.FastaIO as FIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from pyteomics import mgf, mzml
from typing import Iterator, List, Optional, Tuple
from pathlib import Path
import copy
import h5py as h5
import logging
import itertools
import pandas as pd 
from line_profiler import profile



def tensorize_spectrum(spectrum, max_mz, max_peaks):
    """tensorize_spectrum _summary_
    Args:
        mz_array (_type_): _description_
        intensities (_type_): _description_
    Returns:
        _type_: _description_
    """
    idx = spectrum['m/z array'] < max_mz # & (spectrum['intensity array'] > 0)
    _mz_array = spectrum['m/z array'][idx]
    _intensities = spectrum['intensity array'][idx]
    _intensities /= _intensities.max()
    n_peaks = len(_mz_array)
    #if n_peaks <=self.min_peaks:
    #    return None 
    assert len(_intensities) == n_peaks
    # select peaks if exceeds max_n_peaks
    if n_peaks > max_peaks:
        indexes = sorted(range(0, n_peaks),
                         key=lambda x: spectrum['intensity array'][x])[-max_peaks:]
        indexes = sorted(indexes)
        _mz_array = _mz_array[indexes]
        _intensities = _intensities[indexes]
        assert len(_mz_array) == max_peaks
    spectrum = np.array([_mz_array, _intensities], dtype=np.float32)
    spectrum = np.pad(spectrum, ((0, 0), (0, max_peaks - len(_mz_array))),
                      mode="constant", constant_values=0)
    return spectrum



# TODO: add multi-worker support
class SpectraStream(IterableDataset):
    """SpectraStream SpectraStream used during search, currently support mgf only 

    """
    def __init__(self, mgfs_path: Path, 
                 min_mz: float = 0., 
                 max_mz: float = 2000., 
                 min_peaks: int = 64,
                 max_peaks: int = 512,
                 max_charge: int = 6) -> None:
        """__init__ 

        Args:
            mgfs_path (List): _description_
            min_mz (float, optional): _description_. Defaults to 0..
            max_mz (float, optional): _description_. Defaults to 2000..
            min_peaks (int, optional): _description_. Defaults to 64.
            max_peaks (int, optional): _description_. Defaults to 512.
            max_charges (int, optional): _description_. Defaults to 6.
        """
        super().__init__()
        self.mgfs_path = mgfs_path
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.max_charge = max_charge 
    

    def collect_metadata(self, spectrum):
        charge = spectrum['params']['charge'][0]
        pre_mz = spectrum['params']['pepmass'][0]  
        neutral_mass = pre_mz * charge - charge * pp.PROTON - pp.H2O
        rt = spectrum['params']['rtinseconds'] 
        scan = spectrum['params']['scans']
        return {'charge': charge,
                'pre_mz': pre_mz,
                'neutral_mass': neutral_mass,
                'rt': rt,
                'scan': int(scan),
                }


    def __iter__(self):
        with mgf.read(str(self.mgfs_path), convert_arrays=1, read_charges=False) as reader:
            for spectrum in reader:
                if self.min_peaks <= len(spectrum['m/z array']):
                    metadata = self.collect_metadata(spectrum)
                    if metadata['charge'] <= self.max_charge:
                        t_spectrum = tensorize_spectrum(spectrum, self.max_mz, self.max_peaks)
                        t_spectrum = torch.from_numpy(t_spectrum).permute(1, 0)
                        if t_spectrum is None:
                            continue
                        mask = ~(t_spectrum.sum(dim=1).bool())
                        
                        yield {**metadata, 
                               'spectrum': t_spectrum,
                               'mask': mask,
                               }

#TODO: change the naming of mgfs_path/mgf_path since we are supporting mzml now
class SpectraStreamInMemory(Dataset):
    def __init__(self, mgfs_path: Path,
                 min_mz: float = 0.,
                 max_mz: float = 2000.,
                 min_peaks: int = 64,
                 max_peaks: int = 512,
                 max_charge: int = 6,
                 clean_cache: bool = False) -> None:
        """__init__ 

        Args:
            mgfs_path (Path): _description_
            min_mz (float, optional): _description_. Defaults to 0..
            max_mz (float, optional): _description_. Defaults to 2000..
            min_peaks (int, optional): _description_. Defaults to 64.
            max_peaks (int, optional): _description_. Defaults to 512.
            max_charges (int, optional): _description_. Defaults to 6.
        """
        super().__init__()
        self.mgfs_path = mgfs_path
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.max_charge = max_charge
        self.format = self.mgfs_path.suffix[1:].lower()
        self.spectra = [] 
        self.metadata = {
            'charge': [],
            'pre_mz': [],
            'neutral_mass': [],
            'rt': [],
            'scan': [],
        }

        self.__load_mgf()

    
    def collect_metadata(self, spectrum):
        try:
            if self.format == 'mgf':
                charge = spectrum['params']['charge'][0]
                pre_mz = spectrum['params']['pepmass'][0]  
                
                rt = spectrum['params']['rtinseconds'] 
                scan = spectrum['params']['scans']
            elif self.format == 'mzml':
                charge = spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
                pre_mz = float(spectrum['precursorList']['precursor'][0]
                               ['selectedIonList']['selectedIon'][0]['selected ion m/z'])
                rt = float(spectrum['scanList']['scan'][0]['scan start time'])
                scan = spectrum['id'].split('=')[-1]
            else:
                raise NotImplementedError(f"{self.format} is not supported.")
        except:
            raise KeyError(f"Error in parsing {self.format} file.")
        
        neutral_mass = pre_mz * charge - charge * pp.PROTON - pp.H2O
        return {'charge': charge,
                'pre_mz': pre_mz,
                'neutral_mass': neutral_mass,
                'rt': rt,
                'scan': int(scan),
                }
    

    def __load_mgf(self):
        if self.format == 'mgf':
            reader = mgf.read(str(self.mgfs_path), convert_arrays=1, read_charges=False)
        elif self.format == 'mzml':
            reader = mzml.read(str(self.mgfs_path))
        
        for spectrum in reader:
            if self.min_peaks <= len(spectrum['m/z array']):
                metadata = self.collect_metadata(spectrum)
                if metadata['charge'] <= self.max_charge:
                    t_spectrum = tensorize_spectrum(spectrum, self.max_mz, self.max_peaks)
                    if t_spectrum is None:
                        continue
                    t_spectrum = torch.from_numpy(t_spectrum).permute(1, 0)
                    metadata = self.collect_metadata(spectrum)
                    self.spectra.append(t_spectrum)
                    for k, v in metadata.items():
                        self.metadata[k].append(v)

        reader.close() 

    def __len__(self):
        return len(self.spectra)
    

    def __getitem__(self, index):
        mask = ~(self.spectra[index].sum(dim=1).bool())
        return {'spectrum': self.spectra[index],
                'charge': self.metadata['charge'][index],
                'pre_mz': self.metadata['pre_mz'][index],
                'neutral_mass': self.metadata['neutral_mass'][index],
                'rt': self.metadata['rt'][index],
                'scan': self.metadata['scan'][index],
                'mask': mask}

class SpectraStreamH5(Dataset):
    def __init__(self, mgfs_path: Path, 
             min_mz: float = 0., 
             max_mz: float = 2000., 
             min_peaks: int = 64,
             max_peaks: int = 512,
             max_charge: int = 6,
             clean_cache: bool = False) -> None:
        """__init__ 

        Args:
            mgfs_path (List): _description_
            min_mz (float, optional): _description_. Defaults to 0..
            max_mz (float, optional): _description_. Defaults to 2000..
            min_peaks (int, optional): _description_. Defaults to 64.
            max_peaks (int, optional): _description_. Defaults to 512.
            max_charges (int, optional): _description_. Defaults to 6.
        """
        super().__init__()
        self.mgfs_path = mgfs_path
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.max_charge = max_charge 
        self.scan2idx = dict()

        self.n_spectra = self.__load_info()
        self.clean_cache = clean_cache
        self.h5_path = mgfs_path.parent / (mgfs_path.stem + '.h5')
        self.__create_h5(self.h5_path)
        self.h5f = h5.File(self.h5_path, 'r', libver='latest')

    def __load_info(self):
        count = 0
        
        with mgf.read(str(self.mgfs_path), convert_arrays=1, read_charges=False) as reader:
            for spectrum in reader:
                if self.min_peaks <= len(spectrum['m/z array']):
                    if spectrum['params']['charge'][0] <= self.max_charge:
                        self.scan2idx[int(spectrum['params']['scans'])] = count
                        count += 1
        return count
    

    def __create_h5(self, emds_path: Path) -> h5.File:
        if emds_path.exists():
            logging.info(f"{emds_path} already exists, using cached emds.")
            return
        idx = 0
        h5f = h5.File(str(emds_path), 'w', libver='latest')
        h5f.create_dataset('spectra', (self.n_spectra, self.max_peaks * 2), dtype=np.float32)
        h5f.create_dataset('scan', (self.n_spectra), dtype=int)
        h5f.create_dataset('charge', (self.n_spectra), dtype=int)
        h5f.create_dataset('pre_mz', (self.n_spectra), dtype=np.float32)
        h5f.create_dataset('rt', (self.n_spectra), dtype=np.float32) 

        with mgf.read(str(self.mgfs_path), convert_arrays=1, read_charges=False) as reader:
            for spectrum in reader:
                if self.min_peaks <= len(spectrum['m/z array']):
                    if spectrum['params']['charge'][0] <= self.max_charge:
                        t_spectrum = tensorize_spectrum(spectrum, self.max_mz, self.max_peaks)
                        t_spectrum = t_spectrum.reshape(1, -1).squeeze(0)
                        scan = int(spectrum['params']['scans'])
                        h5f['spectra'][idx] = t_spectrum
                        h5f['scan'][idx] = scan
                        h5f['charge'][idx] = spectrum['params']['charge'][0]
                        h5f['pre_mz'][idx] = spectrum['params']['pepmass'][0]
                        h5f['rt'][idx] = spectrum['params']['rtinseconds']
                        idx += 1
        h5f.close()
    

    def __getitem__(self, index):
        spectrum = self.h5f['spectra'][index]
        scan = self.h5f['scan'][index]
        charge = self.h5f['charge'][index]
        pre_mz = self.h5f['pre_mz'][index]
        rt = self.h5f['rt'][index] 
        neutral_mass = pre_mz * charge - charge * pp.PROTON - pp.H2O

        spectrum = spectrum.reshape(2, -1)
        spectrum = torch.from_numpy(spectrum).permute(1, 0)
        mask = ~(spectrum.sum(dim=1).bool())
        return {'spectrum': spectrum, 
                'mask': mask, 
                'scan': scan,
                'charge': charge,
                'pre_mz': pre_mz,
                'rt': rt,
                'neutral_mass': neutral_mass}


    def __len__(self):
        return self.n_spectra
    
    def __del__(self):
        self.h5f.close()
        if self.clean_cache:
            self.h5_path.unlink()
            


#===================================================================================================
class PeptideStream(IterableDataset):
    """PeptideStream Canonical peptide stream used during search, for loading from fasta file, does not support PTM 

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, fasta_path: str, 
                 min_pep_len:int = 6,
                 max_pep_len:int = 32,
                 min_pep_mass: float = 400.,
                 max_pep_mass: float = 6000.,) -> None:
        super().__init__()
        self.min_pep_len = min_pep_len
        self.max_pep_len = max_pep_len
        self.min_pep_mass = min_pep_mass
        self.max_pep_mass = max_pep_mass 
        self.fasta_path = fasta_path
        self.tokenizer = Tokenizer() 
    

    def __iter__(self) -> Iterator:
        with open(self.fasta_path, 'r') as f:
            dec = ''
            for line in f:
                if line[0] == '>':
                    dec = line[1:].strip()
                else:
                    peptide = line.strip()
                
                    if self.min_pep_len <= len(peptide) <= self.max_pep_len:
                        try:
                            pep_mass = pp.cal_pep_mass(peptide)
                        except KeyError:
                            continue
                        if self.min_pep_mass <= pep_mass <= self.max_pep_mass:
                            t_peptide = self.tokenizer.encode(peptide) 

                            yield {'t_peptide': t_peptide, 
                                   'peptide':peptide, 
                                   'pep_mass': pep_mass,
                                   'dec': dec}


def PTM_collate_fn(peptide_batch):
    batch = {k: [dic[k] for dic in peptide_batch] for k in peptide_batch[0]}
    del peptide_batch
    batch['t_peptide'] = torch.stack(batch['t_peptide'])
    if 'pair_bias' in batch:
        batch['pair_bias'] = torch.stack(batch['pair_bias'])
    return batch

class PTMPeptideStream(IterableDataset):
    """PTMPeptideStream PTM peptide stream used during search, for loading from fasta file, support PTM 

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, fasta_path: str, 
                 min_pep_len:int = 6,
                 max_pep_len:int = 32,
                 fix_modifications: List = ['CAM'],
                 var_modifications: Optional[List] = None, 
                 modification_token_id: int = 23, 
                 max_modification: int = 3,
                 decoy_prefix: str = 'DECOY_',
                 return_tensor:bool = True,
                 min_charge = 2,
                 max_charge = 4) -> None:
        super().__init__()
        self.min_pep_len = min_pep_len
        self.max_pep_len = max_pep_len
        self.fasta_path = fasta_path
        self.fix_modifications = fix_modifications
        self.var_modifications = var_modifications 
        self.tokenizer = Tokenizer() 
        self.max_modification = max_modification
        self.modification_token_id = modification_token_id
        self.return_tensor = return_tensor
        self.min_charge = min_charge
        self.max_charge = max_charge
        
        if self.var_modifications is None:
            self.var_modifications = []
        if len(
            self.fix_modifications) != 1 or 'CAM' not in self.fix_modifications:
            raise NotImplementedError("Carbamidomethylation on Cysteine should be specified and is the only fixed modification supported currently.")
        
        for mod in self.var_modifications:
            if mod not in pp.PTM:
                raise NotImplementedError(f"{mod} is not supported currently.")
        
        self.decoy_prefix = decoy_prefix
        if return_tensor:
            self.size, self.n_peptides, self.n_decoys = self.__size()
        
        

    def enumerate_ptm(self, peptide:str):
        yield from pp.enumerate_ptm(peptide, self.var_modifications, self.max_modification)


    def __iter__(self) -> Iterator:
        with open(self.fasta_path, 'r') as f:
            dec = ''
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    dec = line[1:]
                else:
                    peptide = line
                    if self.min_pep_len <= len(peptide) <= self.max_pep_len:
                        try: 
                            pep_mass = pp.cal_pep_mass(peptide)
                        except KeyError:
                            continue
                        
                        for PTM_enum in self.enumerate_ptm(peptide):
                            mod_peptide = list(peptide)
                            mod_mass = 0.
                            for idx, mod in PTM_enum:
                                mod_peptide[idx] = mod_peptide[idx] + \
                                    '(' + mod + ')'    
                                mod_mass += pp.PTM[mod]
                            
                            mod_peptide = ''.join(mod_peptide)
                            t_peptide = None
                            pair_bias = None
                            if self.return_tensor:
                                pair_bias =  cal_pair_bias(
                                    self.max_charge, peptide,  mods=PTM_enum)
                                pair_bias = torch.from_numpy(pair_bias)
                                t_peptide = self.tokenizer.encode(peptide)
                            yield {'mod_peptide': mod_peptide,
                                   'mods': PTM_enum,
                                   'peptide':peptide,
                                   'pep_mass': pep_mass + mod_mass,
                                   't_peptide': t_peptide,
                                   'dec': dec,
                                   'pair_bias': pair_bias,
                                   }
   
    def __len__(self):
        return self.size
    

    def __size(self):
        count = 0 
        pep_count = 0
        decoy_count = 0

        with open(self.fasta_path, 'r') as f:
            dec = ''
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    dec = line[1:]
                else:
                    peptide = line
                    if self.min_pep_len <= len(peptide) <= self.max_pep_len:
                        try: 
                            pep_mass = pp.cal_pep_mass(peptide)
                        except KeyError:
                            continue
                        if dec.startswith(self.decoy_prefix):
                            decoy_count += 1
                        else:
                            pep_count += 1
                        for PTM_enum in self.enumerate_ptm(peptide):
                            count += 1
        return count, pep_count, decoy_count

class PTMPeptideDataset(Dataset):
    def __init__(self, fasta_path: str,
                 min_pep_len: int = 6,
                 max_pep_len: int = 32,
                 fix_modifications: List = ['CAM'],
                 var_modifications: Optional[List] = None,
                 modification_token_id: int = 23,
                 max_modification: int = 3,
                 decoy_prefix: str = 'DECOY_',
                 return_tensor: bool = True,
                 min_charge = 2,
                 max_charge = 4) -> None:
        super().__init__()
        self.min_pep_len = min_pep_len
        self.max_pep_len = max_pep_len
        self.fasta_path = fasta_path
        self.fix_modifications = fix_modifications
        self.var_modifications = var_modifications
        self.tokenizer = Tokenizer()
        self.max_modification = max_modification
        self.modification_token_id = modification_token_id
        self.return_tensor = return_tensor
        self.data = None
        self.min_charge = min_charge
        self.max_charge = max_charge
        if self.var_modifications is None:
            self.var_modifications = []
        if len(
                self.fix_modifications) != 1 or 'CAM' not in self.fix_modifications:
            raise NotImplementedError(
                "Carbamidomethylation on Cysteine should be specified and is the only fixed modification supported currently.")

        for mod in self.var_modifications:
            if mod not in pp.PTM:
                raise NotImplementedError(f"{mod} is not supported currently.")

        self.decoy_prefix = decoy_prefix
        if return_tensor:
            self.size, self.n_peptides, self.n_decoys = self.__size()

    def enumerate_ptm(self, peptide: str):
        yield from pp.enumerate_ptm(peptide, self.var_modifications, self.max_modification)


    def __getitem__(self, index):
        data = self.data.iloc[index]
        pair_bias = cal_pair_bias(
                        self.max_charge, data['peptide'],  mods=data['mods'])
        pair_bias = torch.from_numpy(pair_bias)
        return {
            'mod_peptide': data['mod_peptide'],
            'mods': data['mods'],
            'peptide': data['peptide'],
            'pep_mass': data['pep_mass'],
            'dec': data['dec'],
            't_peptide': self.tokenizer.encode(data['peptide']),
            'pair_bias': pair_bias,
        }
        


    def __len__(self):
        return self.size
    
    def __size(self):
        count = 0
        pep_count = 0
        decoy_count = 0
        mod_peptides = []
        pep_masses = []
        decs = []
        peptides = []
        mods = []
        with open(self.fasta_path, 'r') as f:
            dec = ''
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    dec = line[1:]
                else:
                    peptide = line
                    if self.min_pep_len <= len(peptide) <= self.max_pep_len:
                        try:
                            pep_mass = pp.cal_pep_mass(peptide)
                        except KeyError:
                            continue
                        if dec.startswith(self.decoy_prefix):
                            decoy_count += 1
                        else:
                            pep_count += 1
                        for PTM_enum in self.enumerate_ptm(peptide):
                            count += 1
                            mod_peptide = list(peptide)
                            mod_mass = 0.
                            for idx, mod in PTM_enum:
                               mod_peptide[idx] = mod_peptide[idx] + \
                                   '(' + mod + ')'
                               mod_mass += pp.PTM[mod]
                        
                            # t_peptide = self.tokenizer.encode(peptide)
                            mod_peptide = ''.join(mod_peptide)
                            mod_peptides.append(mod_peptide)
                            mods.append(PTM_enum)
                            pep_masses.append(pep_mass + mod_mass)
                            peptides.append(peptide)
                            decs.append(dec)

        self.data = pd.DataFrame({'mod_peptide': mod_peptides,
                    'mods': mods,
                    'peptide': peptides,
                    'pep_mass': pep_masses,
                    'dec': decs})
        self.data = self.data.sort_values('pep_mass').reset_index(drop=True)
        self.data['rounded_mass'] = self.data['pep_mass'].apply(lambda x: str(round(x, 1)))
        return count, pep_count, decoy_count
    

class PTMPeptideStreamBias(PTMPeptideStream):
    def __init__(self, fasta_path: str, 
                 min_pep_len:int = 6,
                 max_pep_len:int = 32,
                 fix_modifications: List = ['CAM'],
                 var_modifications: Optional[List] = None, 
                 modification_token_id: int = 23, 
                 max_modification: int = 3,
                 max_charge:int = 4,
                 min_charge:int = 2,
                 decoy_prefix: str='DECOY_') -> None:
        super().__init__(fasta_path, min_pep_len, max_pep_len, fix_modifications, var_modifications, modification_token_id, max_modification, decoy_prefix)
        self.max_charge = max_charge
        self.min_charge = min_charge
        self.size = self.size * (self.max_charge - self.min_charge + 1)

    
    def __iter__(self) -> Iterator:
        for charge in range(self.min_charge, self.max_charge + 1):
            for peptide_info in super().__iter__():
                pair_bias = cal_pair_bias(
                    charge, peptide_info['peptide'],  mods=peptide_info['mods'])
                pair_bias = torch.from_numpy(pair_bias)
                yield {**peptide_info, 'charge': charge, 'pair_bias': pair_bias}
        #for peptide_info in super().__iter__():
        #    pair_bias = cal_pair_bias(
        #        self.max_charge, peptide_info['peptide'],  mods=peptide_info['mods'])
        #    for charge in range(2, self.max_charge + 1):
        #        pair_bias_ = np.zeros_like(pair_bias)
        #        pair_bias_[:, :, 0: 6*charge] = pair_bias[:, :, 0: 6*charge]
        #        pair_bias_ = torch.from_numpy(pair_bias_)
        #        yield {**peptide_info, 'charge': charge, 'pair_bias': pair_bias_}
    
    

class PTMPeptideDatasetBias(PTMPeptideDataset):
    def __init__(self, fasta_path: str,
                 min_pep_len: int = 6,
                 max_pep_len: int = 32,
                 fix_modifications: List = ['CAM'],
                 var_modifications: Optional[List] = None,
                 modification_token_id: int = 23,
                 max_modification: int = 3,
                 max_charge: int = 4,
                 min_charge: int = 2,
                 decoy_prefix: str = 'DECOY_') -> None:
        super().__init__(fasta_path, min_pep_len, max_pep_len, fix_modifications,
                         var_modifications, modification_token_id, max_modification, decoy_prefix)
        self.max_charge = max_charge
        self.min_charge = min_charge
        self.size = self.size * (self.max_charge - self.min_charge + 1)

    
    # index is based on charge, then peptide mass
    def __getitem__(self, index):
        i = index // (self.max_charge - self.min_charge + 1)
        data = super().__getitem__(i)
        charge = index % (self.max_charge - self.min_charge + 1) + self.min_charge
        pair_bias = cal_pair_bias(
            charge, data['peptide'],  mods=data['mods'])
        pair_bias = torch.from_numpy(pair_bias)
        return {**data, 'charge': charge, 'pair_bias': pair_bias}


    def __len__(self):
        return self.size 