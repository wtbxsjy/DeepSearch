import torch
from torch.utils.data import DataLoader

from DeepSearch.search.dataset import *
from DeepSearch.search.search_utils import *

import pandas as pd 
from typing import List
from pathlib import Path
import logging
from tqdm import tqdm
import h5py as h5
import pickle



def compute_spectrum_emds(model,
                          spectrum_loader: DataLoader,
                          token_path: Path = None,
                          store_tokens: bool = False,
                          device: torch.device = 'cuda',
                          dtype: torch.dtype = torch.float16,
                          in_memory: bool = False,
                          ) -> Dict[str, List]:

    # notice that when running on CPU, only float32 and bfloat16 are supported
    if device == 'cpu':
        assert dtype == torch.float32 or dtype == torch.bfloat16, \
            'Only support float32 and bfloat16 on cpu'
    
    model.eval()
    if store_tokens:
        #TODO: change hard coded shape
        if not in_memory:
            token_mmap = np.memmap(token_path,
                               dtype='float16', mode='w+', shape=(len(spectrum_loader.dataset), 512, 768))
        else:
            token_mmap = np.zeros((len(spectrum_loader.dataset), 512, 768), dtype='float16')
    
    masks = []
    spectra_emds = [] 
    meta = {
        'scan': [],
        'charge': [],
        'pre_mz': [],
        'rt': [],
        'neutral_mass': []
    }

    offset = 0
    for batch in tqdm(spectrum_loader):
        
        spectra = batch['spectrum'].to(device, non_blocking=True)
        batch_size = spectra.shape[0]
        mask = batch['mask'].to(device, non_blocking=True)
        meta['scan'].extend( batch['scan'].tolist())
        meta['charge'].extend(batch['charge'].tolist())
        meta['pre_mz'].extend(batch['pre_mz'].tolist())
        meta['rt'].extend(batch['rt'].tolist())
        meta['neutral_mass'].extend(batch['neutral_mass'].tolist())

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            spectra_emd, spectra_token = model._encode_spectra(spectra, 
                                                               mask, 
                                                               normalize=True)
            spectra_emd = spectra_emd.detach().cpu().to(torch.float16)
            spectra_emds.extend(list(spectra_emd))
            

            if store_tokens:
                spectra_token = spectra_token.detach().cpu()

                token_mmap[offset:offset+batch_size] = spectra_token.numpy().astype('float16')
                if not in_memory:
                    token_mmap.flush()
                mask = mask.detach().cpu() 
                masks.extend(list(mask))
                offset += batch_size
    if store_tokens:
        return meta, spectra_emds, token_mmap, masks
                             
    else:
        return meta, spectra_emds, None, None


def mgf2emds(model,
             mgf_path: Path,
             emds_dir: Path, 
             device: torch.device, 
             batch_size: int = 512,
             store_tokens: bool = False,
             dtype: torch.dtype = torch.float16,
             n_workers: int = 16,
             in_memory: bool = False,
             **kwargs
             ) :

    
    name = mgf_path.stem
    
    spectra_stream = SpectraStreamInMemory(mgf_path, **kwargs)
    spectrum_loader = DataLoader(spectra_stream, batch_size=batch_size, num_workers=n_workers, pin_memory=True, shuffle=False)
    spectra_info, spectra_emds, spectra_tokens, masks = compute_spectrum_emds(
        model, spectrum_loader, emds_dir/(mgf_path.stem + '.npy'), store_tokens=store_tokens, device=device, dtype=dtype, in_memory=in_memory)
    
    return spectra_info, spectra_emds, spectra_tokens, masks

#==================================================================================================================================


def compute_peptide_emds(model, 
                         emds_dir: Path,
                         file_name: str,
                         db_size: int,
                         device: torch.device,
                         peptide_loader: DataLoader,
                         dtype: torch.dtype = torch.float16,
                         min_charge: int = 2,
                         max_charge: int = 4,
                         **kwargs,
                         ):
    """compute_peptide_emds _summary_

    Args:
        model (_type_): _description_
        emds_dir (Path): _description_
        h5f (h5.File): _description_
        peptide_loader (DataLoader): _description_
        decoy_prefix (str, optional): _description_. Defaults to 'DECOY_'.
        dtype (torch.dtype, optional): _description_. Defaults to torch.float32.
        chunk_size (int, optional): _description_. Defaults to 1e5.`
        max_charge (int, optional): _description_. Defaults to 6.

        h5f: -- peptide 
             -- mod_peptide
             -- pep_mass
             -- dec 
    """
    model.eval()
    

    # now compute peptide embeddings
    offset = 0
    all_meta = peptide_loader.dataset.data
    pep_bucket = {}
    bucket_idx_range = {}
    pep2idx = dict(zip(all_meta['mod_peptide'], all_meta.index))
    #print(len(all_meta['rounded_mass'].unique()))
    logging.info('Building peptide bucket...')
    gb = all_meta.groupby('rounded_mass')
    for rounded_mass in tqdm(all_meta['rounded_mass'].unique()):
        # order is preserved
        df = gb.get_group(rounded_mass)[['mod_peptide', 'pep_mass']]
        pep_bucket[rounded_mass] = list(zip(df['mod_peptide'], df['pep_mass']))
        bucket_idx_range[rounded_mass] = list(df.index)
    
    ss = (max_charge - min_charge + 1) * db_size 
    all_emds = np.memmap(emds_dir/(file_name + '.npy'), dtype='float16', mode='w+', shape=(ss, 512))
    masks = torch.zeros((max_charge - min_charge + 1, 1, 1, 1, 36))
    for i in range(0, max_charge - min_charge +1):
        masks[i, :, :, :, 0: (i+min_charge)*6] = 1
    masks = masks.bool().to(device, non_blocking=True)

    logging.info('Start computing peptide embeddings database')
    for batch in tqdm(peptide_loader):
        t_peptide = batch['t_peptide'].to(device, non_blocking=True)
        t_pair_bias = batch['pair_bias'].to(device, non_blocking=True)
        peptides = batch['peptide']
        batch_size = len(peptides)
        
        # compute pair bias 
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            for charge in range(min_charge, max_charge+1):
                t_pair_bias_ = torch.where(masks[charge-min_charge, :, :, :, :], t_pair_bias, 0)
                pep_emd = model._encode_peptide(t_peptide[:, :-1], 
                                                 attn_bias=t_pair_bias_, 
                                                 normalize=True)[0]
                pep_emd = pep_emd.detach().cpu()
                s = (charge - min_charge) * db_size 
                all_emds[s + offset: s + offset+batch_size] = pep_emd.numpy().astype('float16')
        offset += batch_size
    #assert len(all_emds) == len(pep2idx) * (max_charge - min_charge + 1)
    
    logging.info(
        f'Finished computing peptide embeddings database')
    # save everything
    all_meta = all_meta.to_dict(orient='list')
    save_bucket(emds_dir/(file_name+ '.pkl'), pep_bucket, pep2idx, bucket_idx_range, all_meta)
    logging.info(f'Embedding database saved to {emds_dir}')
    return all_meta, pep_bucket, pep2idx, bucket_idx_range

    
