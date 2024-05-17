import torch
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
from DeepSearch.search.dataset import *
from DeepSearch.search.compute_embedding import *
from DeepSearch.search.search_utils import *
import DeepSearch.utils.peptide as pp
import os

import pandas as pd
from pyteomics import mgf 
import h5py as h5
from tqdm import tqdm
from line_profiler import profile
from functools import partial
from DeepSearch.utils.tokenizer import Tokenizer



def output_result(ostream, hits: Dict):
    # scan, precursor_mz, charge, peptide, ppm_error, score, protein, decoy
    try:
        ostream.write(f"{hits['scan']}\t{hits['pre_mz']}\t{hits['charge']}\t{hits['peptide']}\t{hits['ppm_error']}\t{hits['score']}\n")
    except KeyError:
        ostream.write(f"{hits['scan']}\t{hits['pre_mz']}\t{hits['charge']}\t{'NA'}\t{'NA'}\t{'NA'}\n")
    
def ppm_filtering(peptide_candidates, pre_mz: float, charge:int, precursor_ppm: float):
    candidates_ = [] 
    candidate_ppm_ = []
    for peptide, pep_mass in peptide_candidates:
        ppm = cal_precursor_ppm_error(pre_mz, pep_mass, charge)
        if abs(ppm) < precursor_ppm:
            candidates_.append(peptide)
            candidate_ppm_.append(ppm)
        #candidates_.append(peptide)
        #candidate_ppm_.append(ppm)
    return candidates_, candidate_ppm_


def control_FDR(hits: pd.DataFrame, FDR: float):
    hits = hits.replace('NA', pd.NA)
    hits = hits[hits['score'].isna() != True]
    hits = hits.sort_values(by='score', ascending=True).reset_index(drop=True)
    total_hits = len(hits)
    try:
        total_peptide_hit = hits['decoy'].value_counts()['-']
        total_decoy_hit = hits['decoy'].value_counts()['+']
    except KeyError:
        return hits
    assert total_hits == total_peptide_hit + total_decoy_hit 

    curr_peptide_hit = total_peptide_hit
    curr_decoy_hit = total_decoy_hit
    fdr_arr = [0.] * total_hits
    q_arr = [0.] * total_hits
    for i, row in hits.iterrows():
        if curr_peptide_hit == 0 or curr_decoy_hit == 0:
            break
        fdr_i = (curr_decoy_hit) / curr_peptide_hit
        if row['decoy'] == '-':
            curr_peptide_hit -= 1
        else:
            curr_decoy_hit -= 1


        fdr_arr[i] = fdr_i

    #hits['fdr'] = fdr_arr
    
    min_fdr = 10000. 
    control_position = 0
    for i, fdr in enumerate(fdr_arr):
        min_fdr = min(min_fdr, fdr)
        q_arr[i] = min_fdr
        if q_arr[i] >= FDR:
            control_position = i
    #hits['q_value'] = q_arr
    controled_hits = hits.iloc[control_position:]
    return controled_hits


def combine_results(result_dir, mgfs_path: List[Path]):
    hits = []
    for mgf_path in mgfs_path:
        mgf_name = mgf_path.stem
        result = result_dir / f'{mgf_name}.tsv'   
        df = pd.read_csv(result, sep='\t')
        df.insert(0, 'file', mgf_name)
        hits.append(df)
    hits = pd.concat(hits)

    return hits.reset_index(drop=True)


@profile
def search_db(args,
              model,
              mgfs_path: List[Path],
              mgf_emds_dir: Path,
              db_emds_path: Path,
              output_dir: Path,
              peptide_bucket: Optional[Dict] = None,
              pep2idx: Optional[Dict] = None,
              bucket_idx_range: Optional[Dict] = None,
              pep_meta: Optional[Dict] = None,
              dtype: torch.dtype = torch.float16,
              device: torch.device = 'cuda',
              decoy_prefix: str = 'DECOY_',
              precursor_ppm: float = 10.0,
              top_k: int = 1, 
              re_selection: bool = True,
              open_search: bool = False,
              open_search_window: int = 350,
              ):
    min_charge = args.min_charge
    if peptide_bucket is None:
        peptide_bucket, pep2idx, bucket_idx_range, pep_meta = load_bucket(
            db_emds_path.parent / (db_emds_path.stem + '.pkl'))
    meta_size = len(pep2idx)
    db_size = meta_size * (args.max_charge - args.min_charge + 1)
    print(meta_size)
    db_emds = np.array(np.memmap(db_emds_path, dtype='float16', mode='r', shape=(db_size, 512)))
    db_emds = torch.from_numpy(db_emds)
    db_emds.requires_grad = False
    db_emds = db_emds.to(device, non_blocking=True, dtype=torch.float16)
    
    all_hits = []
    if re_selection:
        tokenizer = Tokenizer()

    # load each mgf first
    for mgf_path in mgfs_path:
        mgf_name = mgf_path.stem
        
        logging.info(f'Computing spectra embedding for {mgf_path}')
        # compute mgf embedding
        # spectra_info: dictionary of list
        # spectra_emds: list of tensor
        # spectra_tokens: np.memmap
        # spectra_masks: list of tensor
        spectra_info, spectra_emds, spectra_tokens, spectra_masks = mgf2emds(model, mgf_path, mgf_emds_dir,
                 batch_size=args.spectra_batch_size,
                 store_tokens=re_selection,
                 dtype=dtype,
                 device=args.device,
                 n_workers=args.n_workers,
                 min_peaks=args.min_peaks,
                 max_charge=args.max_charge,
                 min_mz=args.min_mz,
                 max_mz=args.max_mz,
                 in_memory=args.in_memory,
                 )
        logging.info(f'Finished computing spectra embeddings for {mgf_path}.') 
        
        hits = {
            'scan': [],
            'pre_mz': [],
            'charge': [],
            'rt': [], # 'rtinseconds
            'peptide': [],
            'modified_peptide': [],
            'neutral_mass': [],
            'ppm_error': [],
            'score': [],
            'protein': [],
            'decoy': []
        }

        
        logging.info(f'Conducting main search for {mgf_path}')
        n_spectra = len(spectra_emds)
        
        for i_emd in tqdm(range(n_spectra), total=n_spectra):
            scan = spectra_info['scan'][i_emd]
            pre_mz = spectra_info['pre_mz'][i_emd] 
            charge = spectra_info['charge'][i_emd] 
            rt = spectra_info['rt'][i_emd] 
            pre_mass = spectra_info['neutral_mass'][i_emd]

            spectrum_emd = spectra_emds[i_emd].unsqueeze(0).to(
                device, non_blocking=True, dtype=torch.float16)
            if re_selection:
                # TODO: chunk reading support
                spectra_token = torch.from_numpy(spectra_tokens[i_emd]).to(
                    device, non_blocking=True, dtype=torch.float16).unsqueeze(0)
                spectra_mask = spectra_masks[i_emd].unsqueeze(0).to(
                    device, non_blocking=True)
                
            hit = {
                    'scan': scan,
                    'pre_mz': pre_mz,
                    'charge': charge,
                    'modified_peptide': 'NA',
                    'neutral_mass': 'NA',
                    'peptide': 'NA',
                    'ppm_error': 'NA',
                    'protein': 'NA',
                    'score': 'NA',
                    'decoy': 'NA',
                    'rt': rt,
                }
            
            # retrieve peptide candidates from bucket, given precursor mass 
            if not open_search:
                peptide_candidates, candidate_ppms, indices = retrieve_candidates_close(bucket_idx_range, peptide_bucket,
                                                                           charge, pre_mass, pre_mz, precursor_ppm)
            else: 
                peptide_candidates, indices = retrieve_candidates_open(peptide_bucket, bucket_idx_range, pre_mass, open_search_window * 10)
                
            if len(peptide_candidates) == 0:
                for k, v in hit.items():
                    hits[k].append(v)
                continue
            
            indices = torch.from_numpy(indices) + (charge - min_charge) * meta_size
            indices = indices.to(device)
            peptides_emds = torch.index_select(db_emds, 0, indices)
            
            # compute cosine similarity between spectrum and peptide candidates
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                logits = (spectrum_emd @ peptides_emds.t())#.detach().cpu()
                scores, ranking = logits.sort(descending=True)
                scores = scores.detach().cpu()
                ranking = ranking.detach().cpu()
                PSM_indices = ranking[0, :].tolist()
                scores = scores[0, :].tolist()

                n_scores = len(PSM_indices)
                PSM_idx = PSM_indices[0]
                score = scores[0]
                PSM_pep = peptide_candidates[PSM_idx][0]

                # if reselection is True, we will rescore the top 10 candidates
                if re_selection:
                    top_n = 10
                    rescore_peps = []
                    if n_scores < 10:
                        top_n = n_scores
                    top_indices = PSM_indices[:top_n]
                    for idx in top_indices:
                        rescore_peps.append(peptide_candidates[idx][0])
                    rescore_pep_meta = [extract_peptide_meta(pep_meta, pep2idx, pep) for pep in rescore_peps]
                    rescore_pep_meta = {k: [dic[k] for dic in rescore_pep_meta]
                                        
                         for k in rescore_pep_meta[0]}
                    rescore_mods = rescore_pep_meta['mods']
                    rescore_ori_peps = rescore_pep_meta['peptide']

                    t_peptides = list(
                        map(tokenizer.encode, rescore_ori_peps))
                    t_peptides = torch.stack(t_peptides).to(device, non_blocking=True)
                    pair_bias = [torch.from_numpy(cal_pair_bias(charge, ori_pep, mods=mods, neutral_mass=pre_mass + pp.H2O))
                                 for ori_pep, mods in zip(rescore_ori_peps, rescore_mods)]
                    pair_bias = torch.stack(pair_bias).to(device, non_blocking=True)

                    _, rescore_tokens, rescore_bias = model._encode_peptide(t_peptides.to(
                        device)[:, :-1], True, use_mask=True, attn_bias=pair_bias.to(device))
                    aa_logits = model.multimodal_decoder(
                        spectra_token, spectra_mask, rescore_tokens, rescore_bias)
                    rescore_labels = t_peptides[:, 1:]
                    re_score = rescore(
                        aa_logits, rescore_labels, rescore_mods)
                    # now retrive the top PSM
                    idx = argmax(re_score)
                    PSM_idx = PSM_indices[idx]
                    PSM_pep = rescore_peps[idx]
                    assert PSM_pep == peptide_candidates[PSM_idx][0]
                    score = scores[idx]
            
            meta = extract_peptide_meta(pep_meta, pep2idx, peptide_candidates[PSM_idx][0])
            meta.pop('mods')
            decoy = '+' if meta['protein'].startswith(decoy_prefix) else '-'
            if open_search:
                ppm = 0
            else:
                ppm = candidate_ppms[PSM_idx]
            hit = {
                'scan': scan,
                'pre_mz': pre_mz,
                'charge': charge,
                'modified_peptide': peptide_candidates[PSM_idx][0],
                'ppm_error': ppm,
                'score': score,
                'decoy': decoy,
                'rt': rt,
                **meta
            }
            for k, v in hit.items():
                hits[k].append(v)
        hits = pd.DataFrame(hits)
        hits.to_csv(output_dir / f'{mgf_name}.tsv', sep='\t', index=False)
        logging.info(
            f'Finished main search for {mgf_path}, results are stored in {output_dir / f"{mgf_name}.tsv"}')
        hits.insert(0, 'file', mgf_name)
        all_hits.append(hits)
        if re_selection:
            del spectra_masks, spectra_tokens
            t_path = mgf_emds_dir/(mgf_path.stem + '.npy')
            if t_path.exists():
                os.remove(t_path)
        
    logging.info(
        f'Finished main search for all scans, results are stored in {output_dir / f"all_PSM.tsv"}')
    all_hits = pd.concat(all_hits).reset_index(drop=True)
    all_hits.to_csv(output_dir / 'all_PSM.tsv', sep='\t', index=False)
    return all_hits
    

def rescore(logits, labels, mods=None):
    n_hit, t_len = logits.shape[0], logits.shape[1]
    if mods is not None:
        assert n_hit == len(mods)
    seq_quals = []
    for j in range(n_hit):
        prob = F.softmax(logits[j, :], dim=-1)
        label = labels[j, :]
        accum = 0.
        cc = 0
        for i in range(t_len):
            if mods is not None:
                for mod in mods[j]:
                    if i - 1 in mod:
                        continue
            elif label[i] == 0 or label[i] == 24:
                break
            p = 1 - prob[i, label[i]]
            cc += 1
            accum += p
        aver = accum / cc
        seq_quals.append(-10 * math.log10(aver + 1e-7))

    return seq_quals


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


#TODO add support for onfly search
def search_onfly(args,
                 model,
                 mgfs_path: List[Path],
                 mgf_emds_dir: Path,
                 output_dir: Path,
                 peptide_bucket: Optional[Dict] = None,
                 pep_meta: Optional[Dict] = None,
                 dtype: torch.dtype = torch.float16,
                 device: torch.device = 'cuda',
                 decoy_prefix: str = 'DECOY_',
                 precursor_ppm: Optional[float] = 10.0,
                 top_k: int = 1, 
                 re_selection: bool = True,):

    store_tokens = re_selection
    all_hits = []
    tokenizer = Tokenizer()
    # load each mgf first
    for mgf_path in mgfs_path:
        mgf_name = mgf_path.stem

        logging.info(f'Computing spectra embedding for {mgf_path}')
        # compute mgf embedding
        # spectra_info: dictionary of list
        # spectra_emds: list of tensor
        # spectra_tokens: np.memmap
        # spectra_masks: list of tensor
        spectra_info, spectra_emds, spectra_tokens, spectra_masks = mgf2emds(model, mgf_path, mgf_emds_dir,
                                                                             batch_size=args.spectra_batch_size,
                                                                             store_tokens=re_selection,
                                                                             dtype=dtype,
                                                                             device=args.device,
                                                                             n_workers=args.n_workers,
                                                                             min_peaks=args.min_peaks,
                                                                             max_charge=args.max_charge,
                                                                             min_mz=args.min_mz,
                                                                             max_mz=args.max_mz,
                                                                             in_memory=args.in_memory,
                                                                             )
        logging.info(f'Finished computing spectra embeddings for {mgf_path}.')

        hits = {
            'scan': [],
            'pre_mz': [],
            'charge': [],
            'rt': [],  # 'rtinseconds
            'peptide': [],
            'modified_peptide': [],
            'neutral_mass': [],
            'ppm_error': [],
            'score': [],
            'protein': [],
            'decoy': []
        }

        logging.info(f'Conducting main search for {mgf_path}')
        n_spectra = len(spectra_emds)

        for i_emd in tqdm(range(n_spectra), total=n_spectra):
            scan = spectra_info['scan'][i_emd]
            pre_mz = spectra_info['pre_mz'][i_emd]
            charge = spectra_info['charge'][i_emd]
            rt = spectra_info['rt'][i_emd]
            pre_mass = spectra_info['neutral_mass'][i_emd]

            spectrum_emd = spectra_emds[i_emd].unsqueeze(0).to(
                device, non_blocking=True, dtype=torch.float16)
            if re_selection:
                # TODO: chunk reading support
                spectra_token = torch.from_numpy(spectra_tokens[i_emd]).to(
                    device, non_blocking=True, dtype=torch.float16).unsqueeze(0)
                spectra_mask = spectra_masks[i_emd].unsqueeze(0).to(
                    device, non_blocking=True)
            hit = {
                    'scan': scan,
                    'pre_mz': pre_mz,
                    'charge': charge,
                    'modified_peptide': 'NA',
                    'neutral_mass': 'NA',
                    'peptide': 'NA',
                    'ppm_error': 'NA',
                    'protein': 'NA',
                    'score': 'NA',
                    'decoy': 'NA',
                    'rt': rt,
                }
            
            peptide_candidates = retrieve_peptide_candidate(
                peptide_bucket, pre_mass, 1)
            if len(peptide_candidates) == 0:
                for k, v in hit.items():
                    hits[k].append(v)
                continue
            peptide_candidates, candidate_ppm = ppm_filtering(
                peptide_candidates, pre_mz, charge, precursor_ppm)
            if len(peptide_candidates) == 0:
                for k, v in hit.items():
                    hits[k].append(v)
                continue

            candidate_mods, candidate_ori_peps = [], []
            for pep in peptide_candidates:
                candidate_mods.append(pep_meta[pep]['mods'])
                candidate_ori_peps.append(pep_meta[pep]['peptide'])
                
            t_peptides = list(map(tokenizer.encode, candidate_ori_peps))
            t_peptides = torch.stack(t_peptides).to(device, non_blocking=True) 
            pair_bias = [torch.from_numpy(cal_pair_bias(charge, ori_pep, mods=mods, neutral_mass=pre_mass + pp.H2O)) 
                         for ori_pep, mods in zip(candidate_ori_peps, candidate_mods)]
            pair_bias = torch.stack(pair_bias).to(device, non_blocking=True)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                
                candidate_emd, candidate_tokens, candidate_bias = model._encode_peptide(t_peptides.to(
                    device)[:, :-1], True, use_mask=True, attn_bias=pair_bias.to(device))
                candidate_emd = candidate_emd.to(torch.float16)
                logits = (spectrum_emd @ candidate_emd.t()).detach().cpu()
                scores, ranking = logits.sort(descending=True)
                PSM_indices = ranking[0, :].tolist()
                scores = scores[0, :].tolist()
                n_scores = len(PSM_indices)
                PSM_idx = PSM_indices[0]
                score = scores[0]
                PSM_pep = peptide_candidates[PSM_idx]
                
                if re_selection:
                    top_n = 10
                    #rescore_peplen = []
                    rescore_peps = []
                    rescore_mods = []
                    if n_scores < 10:
                        top_n = n_scores
                    top_indices = PSM_indices[:top_n]
                    for idx in top_indices:
                        rescore_peps.append(peptide_candidates[idx])
                        rescore_mods.append(candidate_mods[idx])
                        #rescore_peplen.append(len(peptide_candidates[idx]))
                    top_indices = torch.tensor(top_indices).to(device)
                    rescore_pep_token = torch.index_select(candidate_tokens, 0, top_indices)
                    rescore_labels = torch.index_select(t_peptides, 0, top_indices)[:, 1:]
                    rescore_bias = torch.index_select(candidate_bias, 0, top_indices)
                    aa_logits = model.multimodal_decoder(spectra_token, spectra_mask, rescore_pep_token, rescore_bias)
                    re_score = rescore(
                        aa_logits, rescore_labels, rescore_mods)
                    idx = argmax(re_score)
                    PSM_idx = PSM_indices[idx]
                    PSM_pep = rescore_peps[idx]
                    assert PSM_pep == peptide_candidates[PSM_idx]
                    score = scores[idx]
            


            meta = {
                'neutral_mass': pep_meta[PSM_pep]['pep_mass'],
                'peptide': pep_meta[PSM_pep]['peptide'],
                'protein': pep_meta[PSM_pep]['dec']
            }
            decoy = '+' if meta['protein'].startswith(decoy_prefix) else '-'
            hit = {
                'scan': scan,
                'pre_mz': pre_mz,
                'charge': charge,
                'modified_peptide': peptide_candidates[PSM_idx],
                'ppm_error': candidate_ppm[PSM_idx],
                'score': score,
                'decoy': decoy,
                'rt': rt,
                **meta
            }
            for k, v in hit.items():
                hits[k].append(v)

        hits = pd.DataFrame(hits)
        hits.to_csv(output_dir / f'{mgf_name}.tsv', sep='\t', index=False)
        logging.info(
            f'Finished main search for {mgf_path}, results are stored in {output_dir / f"{mgf_name}.tsv"}')
        hits.insert(0, 'file', mgf_name)
        all_hits.append(hits)
        if re_selection:
            del spectra_masks, spectra_tokens
            t_path = mgf_emds_dir/(mgf_path.stem + '.npy')
            if t_path.exists():
                os.remove(t_path)

    logging.info(
        f'Finished main search for all scans, results are stored in {output_dir / f"all_PSM.tsv"}')
    all_hits = pd.concat(all_hits).reset_index(drop=True)
    all_hits.to_csv(output_dir / 'all_PSM.tsv', sep='\t', index=False)
    return all_hits


