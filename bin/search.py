import logging
import os
import re
import sys
import random
import yaml

import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from DeepSearch.search.params import parse_args, var_modification_id, fix_modification_id
from DeepSearch.search.dataset import *
import DeepSearch.utils.peptide as pp
from DeepSearch.model.models import DeepSearch
from DeepSearch.search.compute_embedding import *
from DeepSearch.search.search import *
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def setup_logger(log_file, level):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name)
               for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def load_model(model_path, config: Dict):
    if Path(model_path).exists():
        model = DeepSearch(**config['Model']['params'])
        dict_ = torch.load(model_path)
        state_dict = OrderedDict([(k[7:], v)
                                 for k, v in dict_['state_dict'].items()])
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model {model_path} not found!")
    return model


def configure_search(args, config):
    if args.max_charge > config['max_charge']:
        args.max_charge = config['max_charge']
        logging.info(f"Max charge capped to {args.max_charge}")
    if args.min_peaks < config['min_npeaks']:
        args.min_peaks = config['min_npeaks']
        logging.info(f"Min peaks in a spectrum capped to {args.min_peaks}")
    if args.max_pep_len > config['max_pep_len']:
        args.max_pep_len = config['max_pep_len']
        logging.info(f"Max peptide length capped to {args.max_pep_len}")
    if args.min_pep_len < config['min_pep_len']:
        args.min_pep_len = config['min_pep_len']
        logging.info(f"Min peptide length capped to {args.min_pep_len}")
    if args.min_mz < config['start_mz']:
        args.min_mz = config['start_mz']
        logging.info(f"Min m/z capped to {args.min_mz}")
    if args.max_mz > config['max_mz']:
        args.max_mz = config['max_mz']
        logging.info(f"Max m/z capped to {args.max_mz}")


def configure_PTM(args):
    if type(args.fix_mod) is not list:
        args.fix_mod = [args.fix_mod]
    if 1 not in args.fix_mod or len(args.fix_mod) != 1:
        raise NotImplementedError(
            "Only Carbamidomethyl on Cysteine is supported as fixed modification!")
    else:
        args.fix_mod = ['CAM']

    var_mod = []
    if args.var_mod is None:
        args.var_mod = []
    elif type(args.var_mod) is not list:
        args.var_mod = [args.var_mod]

    for mod_id in args.var_mod:
        if mod_id not in var_modification_id:
            raise NotImplementedError(
                f"{var_modification_id[mod_id]} is not supported currently.")
        else:
            var_mod.append(var_modification_id[mod_id])
    args.var_mod = var_mod
    logging.info(f"Fixed modification: {args.fix_mod}")
    if not args.open_search:
        logging.info(f"Variable modification: {args.var_mod}")


def build_onfly_peptide_bucket(args):
    configure_PTM(args)
    peptide_bucket, peptide_meta = dict(), dict()
    logging.info(f'Building peptide bucket...')
    pepstream = PTMPeptideStream(str(args.database),
                                 min_pep_len=args.min_pep_len,
                                 max_pep_len=args.max_pep_len,
                                 fix_modifications=args.fix_mod,
                                 var_modifications=args.var_mod,
                                 max_modification=args.max_mod,
                                 return_tensor=False,
                                 decoy_prefix=args.decoy_prefix,
                                 min_charge=args.min_charge,
                                 max_charge=args.max_charge
                                 )
    it = iter(pepstream)
    for pep_info in tqdm(it):
        pep_mass = pep_info['pep_mass']
        round_mass = str(round(pep_mass, 1))
        if round_mass not in peptide_bucket:
            peptide_bucket[round_mass] = [(pep_info['mod_peptide'], pep_mass)]
        else:
            peptide_bucket[round_mass].append(
                (pep_info['mod_peptide'], pep_mass))
        peptide_meta[pep_info['mod_peptide']] = {
            'mods': pep_info['mods'],
            'dec': pep_info['dec'],
            'peptide': pep_info['peptide'],
            'pep_mass': pep_mass,
        }
    return peptide_bucket, peptide_meta


def build_peptide_emds(args, model):
    configure_PTM(args)
    pep2idx, peptide_bucket, peptide_meta = None, None, None
    # build peptide embedding if not specified
    if args.embedding is None:
        if args.compute_embedding:
            logging.info(
                f'Embedding not specified, computing embedding database from {args.database}')
            pepstream = PTMPeptideDataset(str(args.database),
                                          min_pep_len=args.min_pep_len,
                                          max_pep_len=args.max_pep_len,
                                          fix_modifications=args.fix_mod,
                                          var_modifications=args.var_mod,
                                          max_modification=args.max_mod,
                                          max_charge=args.max_charge,
                                          min_charge=args.min_charge,
                                          decoy_prefix=args.decoy_prefix
                                          )

            db_size, pep_count, decoy_count = pepstream.size, pepstream.n_peptides, pepstream.n_decoys
            logging.info(
                f"Qualitfied peptides count: {pep_count + decoy_count}, decoy percentage: {decoy_count / (decoy_count + pep_count) * 100}%.")

            pep_loader = DataLoader(
                pepstream, batch_size=args.peptide_batch_size,
                num_workers=args.n_workers, collate_fn=PTM_collate_fn, shuffle=False, drop_last=False)
            peptide_meta, peptide_bucket, pep2idx, bucket_idx_range = \
                compute_peptide_emds(model,
                                     emds_dir=args.result_dir,
                                     file_name=args.database.stem,
                                     db_size=db_size,
                                     peptide_loader=pep_loader,
                                     dtype=torch.float16,
                                     max_charge=args.max_charge,
                                     min_charge=args.min_charge,
                                     device=args.device
                                     )
            args.embedding = str(args.result_dir / f"{args.database.stem}.npy")
    return pep2idx, peptide_bucket, bucket_idx_range, peptide_meta


def close_search(args, model):
    if args.embedding is None:
        if args.compute_embedding:
            pep2idx, peptide_bucket, bucket_idx_range, peptide_meta = build_peptide_emds(
                args, model)
            args.embedding = Path(args.embedding)

            all_hits = search_db(
                args,
                model,
                mgfs_path=args.input,
                mgf_emds_dir=args.result_dir,
                db_emds_path=args.embedding,
                output_dir=args.result_dir,
                peptide_bucket=peptide_bucket,
                pep2idx=pep2idx,
                bucket_idx_range=bucket_idx_range,
                pep_meta=peptide_meta,
                dtype=torch.float16,
                device=args.device,
                decoy_prefix=args.decoy_prefix,
                precursor_ppm=args.precursor_ppm,
                top_k=1,
                re_selection=args.reselection,
                open_search=args.open_search,
                open_search_window=args.open_search_mass_tol
            )
        else:
            logging.info(f'Embedding not specified, searching on the fly.')
            assert args.open_search == False, "Open search is not supported when embedding is not specified."
            peptide_bucket, peptide_meta = build_onfly_peptide_bucket(args)
            all_hits = search_onfly(
                args,
                model,
                mgfs_path=args.input,
                mgf_emds_dir=args.result_dir,
                output_dir=args.result_dir,
                peptide_bucket=peptide_bucket,
                pep_meta=peptide_meta,
                dtype=torch.float16,
                device=args.device,
                decoy_prefix=args.decoy_prefix,
                precursor_ppm=args.precursor_ppm,
                top_k=1,
                re_selection=args.reselection,
            )
    else:
        logging.info(f'Embedding specified, loading embedding into memory.')
        args.embedding = Path(args.embedding)

        all_hits = search_db(
            args,
            model,
            mgfs_path=args.input,
            mgf_emds_dir=args.result_dir,
            db_emds_path=args.embedding,
            output_dir=args.result_dir,
            peptide_bucket=None,
            pep2idx=None,
            pep_meta=None,
            bucket_idx_range=None,
            dtype=torch.float16,
            device=args.device,
            decoy_prefix=args.decoy_prefix,
            precursor_ppm=args.precursor_ppm,
            top_k=1,
            re_selection=args.reselection,
            open_search=args.open_search,
            open_search_window=args.open_search_mass_tol
        )

    return all_hits


def main(args):
    args = parse_args(args)

    args.device = torch.device(args.device)
    args.result_dir = Path(args.result_dir)
    args.result_dir.mkdir(exist_ok=True)
    args.emds_dir = args.result_dir
    setup_logger(args.result_dir / 'search.log', logging.INFO)

    logging.info(f'Loading model from {args.model} with config {args.config}')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        configure_search(args, config['Meta'])
        # TODO: support top_k >=1
        args.top_k = 1

    model = load_model(args.model, config).to(args.device)
    model.eval()

    #assert args.open_search == False, "Open search is not supported yet!"

    args.input = Path(args.input)
    if args.input.is_dir():
        flist = list(args.input.glob('*.mgf'))
        if len(flist) == 0:
            flist = list(args.input.glob('*.mzML'))
        if len(flist) == 0:
            raise FileNotFoundError(
                f"No mgf or mzML file found in {args.input}!")
        args.input = flist
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input {args.input} not found!")
        args.input = [args.input]

    args.database = Path(args.database)
    if not args.database.exists():
        raise FileNotFoundError(f"Database {args.database} not found!")

    # search
    # if not args.open_search:
    all_hits = close_search(args, model)
    # else:
    #    raise NotImplementedError("Open search is not supported yet!")

    assert args.FDR > 0 and args.FDR <= 1, "FDR should be in (0, 1]"
    if args.FDR < 1:
        controled_hits = control_FDR(all_hits, args.FDR)
        controled_hits.to_csv(
            args.result_dir / 'controled_PSM.tsv', sep='\t', index=False)
        logging.info(
            f'Finished controling FDR, results are stored in {args.result_dir / f"controled_PSM.tsv"}')

    else:
        logging.info("FDR control disabled, no FDR control will be performed.")


if __name__ == "__main__":
    main(sys.argv[1:])
