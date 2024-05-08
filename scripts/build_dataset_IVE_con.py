# save spectra as numpy array in h5 file with its meta info
# split spectra based on the peptides to ensure spectra in train/test/val come from diff peptides
# caculate mean std for m/z and intensity channel for further normalization

import pandas as pd
import h5py as h5
import sys
from pyteomics import mzml, mzxml
from pathlib import Path
import numpy as np
import logging
import random
from typing import List, Dict
from DeepSearch.utils.peptide import AA_MASS
from multiprocessing import Pool
from functools import cmp_to_key
import re
import time
import functools
from tqdm import tqdm


# TODO: change to attribute
"""
h5py
|
|----spectra            vlen dataset
|----peptides           vlen dataset
|----charges            int dataset
|----premzs              float dataset
|----mapping             group
        |-----           pep-index mapping dataset


"""

MASS_TO_AA = {
    '+15.995': 'm',
    '+57.021': 'C',
    '+0.984': None
}


def comparator(pep_a, pep_b):
    mass_a = sum([AA_MASS[_] for _ in pep_a])
    mass_b = sum([AA_MASS[_] for _ in pep_b])
    if mass_a > mass_b:
        return 1
    elif mass_a == mass_b:
        return 0
    else:
        return -1

# Currently only support Ox and Cam


def parse_mod_seq(mod_seq: str) -> str:
    res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', mod_seq.strip())
    res = [r.strip() for r in res if r is not None and r.strip() != '']

    pep = []
    n_frag = len(res)
    for i, frag in enumerate(res):
        if frag[0] == '-' or frag[0] == '+':
            # we drop N-term modification
            if i == 0:
                return None
            # now possible C-term modification
            if i == n_frag - 1:
                return None
            # now drop not implemented mod
            if frag not in MASS_TO_AA:
                return None
            # otherwise continue
            continue
        for aa in frag:
            pep.append(aa)
        if i < n_frag - 1:
            if res[i+1] not in MASS_TO_AA:
                return None
            if aa == 'M':
                if MASS_TO_AA[res[i+1]] == 'm':
                    pep[-1] = 'm'
                elif MASS_TO_AA[res[i+1]] is None:
                    continue
                else:
                    return None

            elif aa == 'C':
                pep[-1] = 'C'
            # proton
            else:
                if MASS_TO_AA[res[i+1]] is None:
                    continue
                else:
                    return None

    return "".join(pep)


def creat_h5f(fname: str, nlines: int) -> h5.File:
    h5f = h5.File(fname, 'w', libver='latest')
    dt_str = h5.special_dtype(vlen=str)
    dt_float = h5.special_dtype(vlen=np.dtype('float32'))

    h5f.create_dataset(
        'spectra', (nlines, ), maxshape=(None, ), dtype=dt_float)
    h5f.create_dataset(
        'peptides', (nlines, ), maxshape=(None, ), dtype=dt_str)
    # dset_files = h5f.create_dataset('rawfiles', (nlines, ), maxshape=(None, ), dtype=dt_str)
    # dset_scans = h5f.create_dataset('scans', (nlines, 1), maxshape=(None, 1), dtype='int32')
    h5f.create_dataset(
        'charges', (nlines, 1), maxshape=(None, 1), dtype='int32')
    h5f.create_dataset(
        'premzs', (nlines, 1), maxshape=(None, 1), dtype='float32')

    h5f.create_group('mapping')
    return h5f


def resize_h5f(h5f: h5.File, nlines: int):
    for key in ['spectra', 'peptides', 'charges', 'premzs']:
        h5f[key].resize(nlines, 0)


def drop_mod(mod_seq: str) -> str:
    regex_non_alpha = re.compile(r'[^A-Za-z]+')
    return regex_non_alpha.sub('', mod_seq)

# Notice, tsv file should be sorted by filename


def process_file(chunk, mgf_dir, min_n_peaks):
    pre_mzs = []
    spectra = []
    mod_seqs = []
    seqs = []
    charges = []

    raw_file = chunk.iloc[0]['filename']
    stream = None 
    raw_file_full = mgf_dir / raw_file

    # open file
    if raw_file.endswith('mzXML'):
            stream = mzxml.MzXML(
                str(raw_file_full), convert_arrays=1, read_charges=False, use_index=True)
    elif raw_file.endswith('mzML'):
        stream = mzml.PreIndexedMzML(
            str(raw_file_full), convert_arrays=1, read_charges=False)
    else:
        raise NotImplementedError
    logging.debug(f'Processing file {raw_file}')


    for _, row in chunk.iterrows():   

        charge = row['charge']

        mod_seq = row['annotation']
        no_mod_seq = row['seq']

        if type(stream) == mzxml.MzXML:
            try:
                # FIXME: we might get wrong psm pairs if scan number and scan index are mismatched
                spectrum = stream.get_by_index(row['scan'] - 1)
                pre_mz = spectrum['precursorMz'][0]['precursorMz']
            except:
                continue
        else:
            spectrum = None
            if type(stream) is mzml.PreIndexedMzML:
                try:
                    spectrum = stream.get_by_id(
                        'controllerType=0 controllerNumber=1 scan=' + str(row['scan']))
                except KeyError:
                    # build our own index
                    stream = mzml.MzML(
                        str(raw_file_full), convert_arrays=1, read_charges=False, use_index=True)
                    spectrum = stream.get_by_index(row['scan'] - 1)
            elif type(stream) is mzml.MzML:
                spectrum = stream.get_by_index(row['scan'] - 1)
            else:
                raise NotImplementedError
            
            if spectrum is None:
                continue
            try:
                precursor_info = spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
            except:
                continue
            try:
                charge_ = precursor_info['charge state']
                if charge_ != charge:
                    continue
            except KeyError:
                print(
                    'File ' + row['filename'] + ' has inconsistent format with scan index ' + str(row['scan'] - 1))
            pre_mz = 0.
            try:
                pre_mz = float(precursor_info['selected ion m/z'])
            except KeyError:
                print(
                    'File ' + row['filename'] + ' has inconsistent format with scan index ' + str(row['scan'] - 1))
                
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
        # use relative intensity?
        intensity = intensity / np.max(intensity)
        # drop spectrum with less peaks
        if len(mz) < min_n_peaks:
            continue
        spectrum_arr = np.concatenate((mz, intensity), axis=None)

        pre_mzs.append(pre_mz)
        spectra.append(spectrum_arr)
        mod_seqs.append(mod_seq)
        charges.append(charge)
        seqs.append(no_mod_seq)

    stream.close()

    return {
        'pre_mz': pre_mzs, 
        'charge': charges, 
        'spectrum': spectra, 
        'seq': seqs, 
        'mod_seq': mod_seqs 
    }




def main():
    if len(sys.argv) != 8:
        print(
            "Usage: build_dataset_IVE.py tsv spectra_dir out min_pep_len max_pep_len min_n_peaks use_mod")
        exit(1)

    prefix = sys.argv[3]
    min_pep_len, max_pep_len = int(sys.argv[-4]), int(sys.argv[-3])
    min_n_peaks = int(sys.argv[-2])
    mgf_dir = Path(sys.argv[2])
    use_mod = bool(int(sys.argv[-1]))

    all_peptides = pd.read_csv(sys.argv[1], sep='\t',
                               usecols=['annotation'])
    nlines = all_peptides.shape[0]
    all_peptides = all_peptides.drop_duplicates()[
        'annotation'].to_list()
    all_peptides = list(set(map(drop_mod, all_peptides)))
    n_peps = len(all_peptides)
    random.Random(89).shuffle(all_peptides)
    logging.info(
        f'In total {n_peps} peptides in inital read')

    fnames = [prefix+'_train.h5', prefix+'_val.h5', prefix+'_test.h5']
    h5fs = list(map(creat_h5f, fnames, [nlines, nlines*0.3, nlines*0.3]))

    # aux var
    count = 0
    n_peaks = 0
    chunk_size = 10 ** 4
    n_peaks_chunk = None

    train_peps, val_peps, test_peps = set(all_peptides[: int(n_peps*0.85)]), set(all_peptides[int(
        n_peps*0.85): int(n_peps*0.95)]), set(all_peptides[int(n_peps*0.95):])

    train_count, val_count, test_count = -1, -1, -1
    train_mapping, val_mapping, test_mapping = dict(), dict(), dict()
    start_time = time.time()
    
    for chunk in pd.read_csv(sys.argv[1], sep='\t', chunksize=chunk_size, low_memory=False):
        n_peaks_chunk = 0

        chunk = chunk[chunk["charge"].apply(lambda x: x <= 6)]
        chunk['seq'] = chunk['annotation'].apply(drop_mod)
        if use_mod:
            chunk['annotation'] = chunk['annotation'].apply(parse_mod_seq)
            chunk = chunk[chunk['annotation'].notna()]
        
        chunk = chunk[chunk['seq'].apply(lambda x: len(
            x) >= min_pep_len and len(x) <= max_pep_len)]
        grouped = None
        try:
            grouped = chunk.groupby('filename')
        except KeyError:
            # empty chunk
            continue
        pool = Pool(processes=8)
        func = functools.partial(process_file, mgf_dir=mgf_dir, min_n_peaks=min_n_peaks)
        results = pool.map(
            func, [chunk for _, chunk in grouped])

        new_results = dict()
        if len(results) == 0:
            continue
        for k in results[0]:
            if k not in new_results:
                new_results[k] = []
            for result in results:
                new_results[k] += result[k]
        results = new_results
    
        n_results = len(results['spectrum'])
        for i in range(n_results):
            if results['seq'][i] in train_peps:
                h5f = h5fs[0]
                train_count += 1
                idx = train_count
                mapping = train_mapping
            elif results['seq'][i] in val_peps:
                h5f = h5fs[1]
                val_count += 1
                idx = val_count
                mapping = val_mapping
            else:
                h5f = h5fs[2]
                test_count += 1
                idx = test_count
                mapping = test_mapping

            h5f['spectra'][idx] = results['spectrum'][i]
            h5f['peptides'][idx] = results['mod_seq'][i]
            h5f['charges'][idx] = results['charge'][i]
            h5f['premzs'][idx] = results['pre_mz'][i]

            # meta split info
            if results['mod_seq'][i] not in mapping:
                mapping[results['mod_seq'][i]] = []
            mapping[results['mod_seq'][i]].append(idx)

            count += 1
            n_peaks_chunk += len(results['spectrum'][i]) / 2

        n_peaks += n_peaks_chunk

    mean_npeaks = n_peaks / count

    logging.info(
        f'Finish reading {count} spectra, average peaks per spectra is {mean_npeaks}')

    for h5f, mapping in zip(h5fs, [train_mapping, val_mapping, test_mapping]):
        for pep, indices in mapping.items():
            h5f['mapping'].create_dataset(
                pep, data=np.array(indices, dtype=np.int32))

    train_count += 1
    test_count += 1
    val_count += 1

    logging.info(
        f'In total {len(train_mapping)} peptides and {train_count} spectra are in the trainning dataset.')
    logging.info(
        f'In total {len(val_mapping)} peptides and {val_count} spectra are in the validation dataset.')
    logging.info(
        f'In total {len(test_mapping)} peptides and {test_count} spectra are in the testing dataset.')

    print(time.time()-start_time)

    for h5f, count in zip(h5fs, [train_count, val_count, test_count]):
        resize_h5f(h5f, count)
        h5f.close()


if __name__ == '__main__':
    logging.basicConfig(filename='../ive_HCD.log', level=logging.INFO)
    main()
