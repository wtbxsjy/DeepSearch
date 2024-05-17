import random
from torch.utils.data.sampler import Sampler
import numpy as np 
import itertools
from DeepSearch.utils.peptide import mass_comparator, cal_pep_mass
from functools import cmp_to_key, partial
from typing import Dict, Iterator, Optional, Sized
from collections import deque


# TODO: change this to ensure every datapoint is used


def shuffle_seq(seq, g, freq=2):
    s_seq = []
    
    for _ in range(freq):
        seq_ = [_ for _ in seq]
        g.shuffle(seq_)
        s_seq.append(''.join(seq_))
    return s_seq


def retrieve_peps(mass: float, db, delta_mass=2):
    candidates = []
    for delta in range(-delta_mass, delta_mass + 1, 1):
        rounded_mass = round(mass + delta, 0)
        if str(rounded_mass) not in db:
            continue
        candidates.extend(db[str(rounded_mass)])

    return candidates


class ContrastivePSMSampler(Sampler[int]):
    def __init__(self, dataset, n_peps_batch, epoch=0, seed=89):
        self.n_pep_batch = n_peps_batch

        self.h5f = dataset.h5f
        self.n_item = self.h5f['spectra'].shape[0]
        self.n_batches = self.n_item // n_peps_batch

        self.peptides = list(self.h5f['mapping'].keys())
        comparator = cmp_to_key(mass_comparator)
        self.peptides.sort(key=comparator)

        self.n_peptides = len(self.peptides)
        assert n_peps_batch < self.n_peptides, "peptide per batch should be much smaller than total peptides in dataset, which is {self.n_peptides}"
        self.epoch = epoch
        self.seed = seed

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.n_batches):
            peptides_idx = g.choice(self.n_peptides, self.n_pep_batch, replace=False)

            batch = []
            for pep_idx in peptides_idx:
                pep_seq = self.peptides[pep_idx]
                batch.append(g.choice(self.h5f['/mapping/' + pep_seq], 1).item())
            yield batch

    def __len__(self):
        return self.n_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


class MassAnchoredContrastivePSMSampler(ContrastivePSMSampler):
    def __init__(self, dataset, n_peps_batch, epoch=0, seed=89):
        super().__init__(dataset, n_peps_batch,  epoch, seed)
        self.curr_mass = 0

    def sample_peptide_idx(self, g):
        anchor_peptide = g.integers(0, self.n_peptides, 1).item()
        # print(anchor_peptide)
        # now we select 2/3 peps around anchor pep, the peps are sorted by their mass
        # the rest 1/3 peps are randomly selected
        # print(anchor_peptide)
        peptide = self.peptides[anchor_peptide]
        self.curr_mass = cal_pep_mass(peptide)
        #n_similar = 2 * self.n_pep_batch // 3
        #n_rand = self.n_pep_batch - n_similar
        #n_sample = n_similar * 2 
        n_sample = self.n_pep_batch * 2
        if anchor_peptide - n_sample // 2 < 0:
            similar_peps_idx = [x for x in range(n_sample)]
        elif anchor_peptide + n_sample // 2 >= self.n_peptides:
            similar_peps_idx = [x for x in range(
                self.n_peptides - n_sample, self.n_peptides, 1)]
        else:
            similar_peps_idx = [x for x in range(
                anchor_peptide - n_sample // 2, anchor_peptide + n_sample // 2, 1)]

        peptides_idx = g.choice(
            similar_peps_idx, self.n_pep_batch, replace=False).tolist()
        g.shuffle(peptides_idx)
        return peptides_idx

    def sample_spectra_idx(self, peptide_idx):
        peptide_idx = list(set(peptide_idx))
        spectra_idx = []
        spectra_idx.extend(list(
            self.h5f['mapping/' + self.peptides[x]]) for x in peptide_idx)
        spectra_idx = list(itertools.chain.from_iterable(spectra_idx))
        return spectra_idx

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            # batch = set()
            peptides_idx = self.sample_peptide_idx(g)
            spectra_idx = self.sample_spectra_idx(peptides_idx)
            n_sample = min(len(spectra_idx), self.batch_size)
            batch = g.choice(spectra_idx, n_sample, replace=False).tolist()
            # for i in range(self.batch_size):
            # pep_seq = self.h5f['peptides'][random.choice(peptides_idx)].decode('utf-8')
            # batch.add(random.choice(self.h5f['/mapping/' + pep_seq]))

            yield batch


class UniqueMassAnchoredContrastivePSMSampler(MassAnchoredContrastivePSMSampler):
    def __init__(self, dataset, batch_size, epoch=0, seed=89):
        super().__init__(dataset, batch_size,  epoch, seed)
    
    def sample_spectra_idx(self, peptide_idx, g):
        peptide_idx = list(set(peptide_idx))
        spectra_idx = []
        for pep_idx in peptide_idx:
            spec_idx = g.choice(self.h5f['mapping/' + self.peptides[pep_idx]], 1).item()
            spectra_idx.append(spec_idx)    
        return spectra_idx

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        
        for _ in range(self.n_batches):
            peptide_idx = self.sample_peptide_idx(g)
            spectra_idx = self.sample_spectra_idx(peptide_idx, g)
            # n_sample = min(len(spectra_idx), self.batch_size)
            yield spectra_idx 


class UniqueMassAnchoredContrastivePSMSamplerWithDecoy(MassAnchoredContrastivePSMSampler):
    def __init__(self, dataset, batch_size, database: Dict, epoch = 0, seed=89, decoy_mult=0.):
        super().__init__(dataset,  batch_size, epoch, seed)
        self.database = database
        self.decoy_seqs = None
        self.decoy_mult = decoy_mult
        
    def sample_spectra_idx(self, peptide_idx, g):
        peptide_idx = list(set(peptide_idx))
        spectra_idx = []
        for pep_idx in peptide_idx:
            spec_idx = g.choice(
                self.h5f['mapping/' + self.peptides[pep_idx]], 1).item()
            spectra_idx.append(spec_idx)
        return spectra_idx

    def sample_peptide_idx(self, g):
        anchor_peptide = g.integers(0, self.n_peptides, 1).item()
        # print(anchor_peptide)
        # now we select 2/3 peps around anchor pep, the peps are sorted by their mass
        # the rest 1/3 peps are randomly selected
        # print(anchor_peptide)
        peptide = self.peptides[anchor_peptide]
        self.curr_mass = cal_pep_mass(peptide)
        #n_similar = 2 * self.n_pep_batch // 3
        #n_rand = self.n_pep_batch - n_similar
        # n_sample = n_similar * 2
        n_sample = self.n_pep_batch * 2
        if anchor_peptide - n_sample // 2 < 0:
            similar_peps_idx = [x for x in range(n_sample)]
        elif anchor_peptide + n_sample // 2 >= self.n_peptides:
            similar_peps_idx = [x for x in range(
                self.n_peptides - n_sample, self.n_peptides, 1)]
        else:
            similar_peps_idx = [x for x in range(
                anchor_peptide - n_sample // 2, anchor_peptide + n_sample // 2, 1)]

        peptides_idx = g.choice(
            similar_peps_idx, self.n_pep_batch, replace=False).tolist()
        #rand_peps_idx = g.choice(
        #    self.n_peptides, n_rand, replace=False).tolist()
        #peptides_idx = rand_peps_idx + peptides_idx
        g.shuffle(peptides_idx)
       

        if self.decoy_mult > 0:
            shuffle_fn = partial(shuffle_seq, freq=self.decoy_mult, g=g)
            sampled_peptides = set([self.peptides[i] for i in peptides_idx])
            self.decoy_seqs = set(retrieve_peps(
                self.curr_mass, self.database, delta_mass=2))

            self.decoy_seqs = list(self.decoy_seqs - sampled_peptides)
            sampled_peptides = list(sampled_peptides)
            sampled_peptides.sort()

            n_decoys = int(len(sampled_peptides) * self.decoy_mult)
            
            if len(self.decoy_seqs) >= n_decoys:
                self.decoy_seqs = list(g.choice(self.decoy_seqs, n_decoys, replace=False))
            else:
                n_to_shuffle = n_decoys - len(self.decoy_seqs)
                self.decoy_seqs = self.decoy_seqs + \
                                list(g.choice(list(itertools.chain.from_iterable(map(shuffle_fn, sampled_peptides))), n_to_shuffle, replace=False))
            
            g.shuffle(self.decoy_seqs)
            
        return peptides_idx


    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.n_batches):
            peptide_idx = self.sample_peptide_idx(g)

            spectra_idx = self.sample_spectra_idx(peptide_idx, g)
            # n_sample = min(len(spectra_idx), self.batch_size)
            yield spectra_idx


class EvaluationSampler():
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.h5f = dataset.h5f
        self.bucket = self.create_bucket()
        self.curr_mass = 0
    
    def create_bucket(self):
        mass = (self.h5f['premzs'][()]* self.h5f['charges'][()]).squeeze(-1)
        mass = mass.tolist()
        bucket = dict()
        for i, m in enumerate(mass):
            rounded = str(round(m, ndigits=0))
            if rounded not in bucket:
                bucket[rounded] = deque()

            bucket[rounded].append(i)
        return bucket
    

    def __iter__(self):    
        for mass in self.bucket.keys():
            self.curr_mass = float(mass)
            indices = self.bucket[mass]
            
            while len(indices) != 0: 
                batch = []
                while True:
                    idx = indices.popleft()
                    batch.append(idx)
                    if len(batch) == self.batch_size or len(indices) == 0:
                        break
                yield batch
                        
                    



