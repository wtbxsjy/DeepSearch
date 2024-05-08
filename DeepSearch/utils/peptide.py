from collections import OrderedDict
from typing import List, Optional, Tuple
from itertools import combinations
from line_profiler import profile

# Atomic elements
PROTON = 1.007276467
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074
P = 30.973762
S = 31.972071

# Common chemical groups 
N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + H * 2
H2O = H * 2 + O
NH3 = N + H * 3

residue_mass = {
    'CO': CO,
    'CHO': CHO,
    'NH2': NH2,
    'H2O': H2O,
    'NH3': NH3
}

PTM = {
    "CAM": 57.0214637236,  # Carbamidomethylation (CAM)
    "OX": 15.99491,  # Oxidation,
    "PHOS": 79.966331, # Phosphorylation,
    "ACET": 42.010565, # Nterm - Acetylation,
}

PTM_RESIDUE = {
    "C": "CAM",
    "M": "OX",
    "S": "PHOS",
    "T": "PHOS",
    "Y": "PHOS",
    "X": "ACET"
}


AA_INDEX = OrderedDict([
    ("<pad>", 0),
    ("<s>", 1),
    ("G", 2),
    ("R", 3),
    ("V", 4),
    ("P", 5),
    ("S", 6),
    ("U", 7),
    ("L", 8),
    ("M", 9),
    ("Q", 10),
    ("N", 11),
    ("Y", 12),
    ("E", 13),
    ("C", 14),
    ("F", 15),
    ("I", 16),
    ("A", 17),
    ("T", 18),
    ("W", 19),
    ("H", 20),
    ("D", 21),
    ("K", 22),
    ('m', 23),
    ("<e>", 24)
])

IUPAC_INDEX = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("m", 30),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])


AA_MASS = {
    "G": 57.021464,
    "R": 156.101111,
    "V": 99.068414,
    "P": 97.052764,
    "S": 87.032028,
    "U": 150.95363,
    "L": 113.084064,
    "M": 131.040485,
    "Q": 128.058578,
    "N": 114.042927,
    "Y": 163.063329,
    "E": 129.042593,
    "C": 103.009185 + PTM["CAM"],
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
}


def gen_PTM_sequence(peptide: str, PTM_list: List[Tuple]):
    """
    Generate PTM sequence from peptide sequence and PTM list
    """
    peptide = list(peptide)
    for (idx, mod) in PTM_list:
        peptide[idx] = peptide[idx] + '(' + mod + ')'
    return peptide


def enumerate_ptm(peptide, mod: Optional[List[str]] = None, max_n_mods=3):
    """enumerate_ptm enumerate all possible PTM sequences from a peptide sequencem,
    return generator of List[char]"""
    aa_index = {
        'S': [],
        'T': [],
        'Y': [],
        'M': [],
    }
    if mod is None:
        mod = []
    for idx, aa in enumerate(peptide):
        if aa in aa_index:
            if idx == len(peptide) - 1 or idx == 0:
                continue
            aa_index[aa].append(idx)

    
    #aa_index['X'] = [0]
    # [idx, mod]
    mod_bags = []
    for aa in aa_index:
        if PTM_RESIDUE[aa] not in mod:
            continue
        for idx in aa_index[aa]:
            mod_bags.append((idx, PTM_RESIDUE[aa]))
    for n_mods in range(0, max_n_mods + 1):
        yield from combinations(mod_bags, n_mods)


def cal_pep_mass(peptide: str, mods: Optional[Tuple[Tuple]] = None):
    """cal_pep_mass Calculate the mass of a peptide"""
    pep_mass = sum([AA_MASS[_] for _ in peptide])  
    if mods is not None:
        for _, mod in mods:
            pep_mass += PTM[mod]
    return  pep_mass


def cal_pep_premz(peptide:str, charge:int, mods: Optional[Tuple[Tuple]] = None):
    mass = cal_pep_mass(peptide, mods)
    premz = (mass + H2O + PROTON * charge) / charge
    return premz


def parse_modification(mod_peptide: str):
    """parse_modification parse modification from modified peptide sequence"""
    mods = []
    peptide = []
    i_aa = 0
    idx = 0
    while idx < len(mod_peptide):
        if mod_peptide[idx] == '(':
            for j in range(idx + 1, len(mod_peptide)):
                if mod_peptide[j] == ')':
                    mods.append((i_aa - 1, mod_peptide[idx + 1:j]))
                    break
            idx = j + 1
        else:
            peptide.append(mod_peptide[idx])
            i_aa += 1
            idx += 1
    return ''.join(peptide), mods

def cal_ion_mz(neutral_mass, charge, ion_type='pre_mz', ion_mod:Optional[str]=None, mods: Optional[Tuple[Tuple]] = None):
    # mass of peptide + H2O
    # precursor 
    if ion_type == 'pre_mz':
        mass = neutral_mass + PROTON * charge
    # b ion
    elif ion_type == 'b':
        mass = neutral_mass + PROTON * charge
    elif ion_type == 'y':
        mass = neutral_mass + H2O + PROTON * charge
    elif ion_type == 'a':
        mass = neutral_mass - CO + PROTON * charge
    else:
        raise NotImplementedError('Mass calculation for ions other than aby is not implemented')
    
    if ion_mod is None:
        pass
    elif ion_mod == 'H2O':
        mass = mass - H2O
    elif ion_mod == 'NH3':
        mass = mass - NH3
    else:
        raise NotImplementedError('Mass calculation for compound loss other than H2O and NH3 is not implemented')

    return mass / charge

 
#def cal_theoretical_fragmentation(peptide, 
#                                  charge: int = 1, 
#                                  ion_type: str = 'b', 
#                                  ion_mod: Optional[str] = None, 
#                                  pre_neutral_mass = None, 
#                                  append_proton: bool = False, 
#                                  append_pep:bool = False,
#                                  max_mz:float = 10000.,
#                                  mods: Optional[Tuple[Tuple]] = None):
#    """cal_theoretical_fragmentation Caution: the maximum neutral_mass must <= 10000.
#
#    Args:
#        peptide (_type_): _description_
#        charge (int, optional): _description_. Defaults to 1.
#        ion_type (str, optional): _description_. Defaults to 'b'.
#        mod (_type_, optional): _description_. Defaults to None.
#        pre_neutral_mass (_type_, optional): _description_. Defaults to None.
#        append_proton (bool, optional): _description_. Defaults to False.
#        append_pep (bool, optional): _description_. Defaults to False.
#        max_mz (float, optional): _description_. Defaults to 2000..
#
#    Returns:
#        _type_: _description_
#    """
#    n_aa = len(peptide)
#    
#    neutral_mass = 0
#    ions_mz = []
#    if pre_neutral_mass is not None:
#        pre_mz = cal_ion_mz(pre_neutral_mass, charge, ion_type=ion_type, ion_mod=ion_mod)
#        ions_mz.append(pre_mz)
#    if ion_type[0] not in 'abc':
#        peptide = peptide[::-1]
#        if mods is not None:
#            mods_dict = {n_aa - 1 - idx: mod for idx, mod in mods}
#    else:
#        if mods is not None: 
#            mods_dict = {idx: mod for idx, mod in mods}
#    
#    for i in range(1, n_aa):
#        aa = peptide[i - 1]
#        neutral_mass += AA_MASS[aa]
#        if mods is not None and i in mods_dict:
#            neutral_mass += PTM[mods_dict[i]]
#        ion_mz = cal_ion_mz(neutral_mass, charge, ion_type, ion_mod=ion_mod)
#        ions_mz.append(ion_mz)
#    if append_pep:
#        aa = peptide[n_aa - 1]
#        neutral_mass += AA_MASS[aa]
#        if mods is not None and n_aa - 1 in mods_dict:
#            neutral_mass += PTM[mods_dict[n_aa-1]]
#        ion_mz = cal_ion_mz(neutral_mass, charge, ion_type, ion_mod=ion_mod)
#        ions_mz.append(ion_mz)
#    if append_proton:
#        ions_mz.append(PROTON )
#    
#    return ions_mz


def cal_theoretical_fragmentation(peptide,
                                   ions_mz: List[float],
                                  charge: int = 1,
                                  ion_type: str = 'b',
                                  ion_mod: Optional[str] = None,
                                  pre_neutral_mass=None,
                                  append_proton: bool = False,
                                  append_pep: bool = False,
                                  max_mz: float = 10000.,
                                  mods: Optional[Tuple[Tuple]] = None):
    """cal_theoretical_fragmentation Caution: the maximum neutral_mass must <= 10000.

    Args:
        peptide (_type_): _description_
        charge (int, optional): _description_. Defaults to 1.
        ion_type (str, optional): _description_. Defaults to 'b'.
        mod (_type_, optional): _description_. Defaults to None.
        pre_neutral_mass (_type_, optional): _description_. Defaults to None.
        append_proton (bool, optional): _description_. Defaults to False.
        append_pep (bool, optional): _description_. Defaults to False.
        max_mz (float, optional): _description_. Defaults to 2000..

    Returns:
        _type_: _description_
    """

    
    neutral_mass = 0
    n_aa = len(peptide)
    idx = 0
    if pre_neutral_mass is not None:
        pre_mz = cal_ion_mz(pre_neutral_mass - H2O, charge,
                            ion_type=ion_type, ion_mod=ion_mod)
        ions_mz[idx] = pre_mz
        idx += 1
    if ion_type[0] not in 'abc':
        peptide = peptide[::-1]
        if mods is not None:
            mods_dict = {n_aa - 1 - i: mod for i, mod in mods}
    else:
        if mods is not None:
            mods_dict = {i: mod for i, mod in mods}

    
    for i in range(1, n_aa):
        aa = peptide[i - 1]
        neutral_mass += AA_MASS[aa]
        if mods is not None and i - 1 in mods_dict:
            neutral_mass += PTM[mods_dict[i - 1]]
        #if ion_type == 'y':
        #    ion_mz = cal_ion_mz(pre_neutral_mass - H2O -  neutral_mass, charge, ion_type, ion_mod=ion_mod, mods=mods)
        ion_mz = cal_ion_mz(neutral_mass, charge, ion_type, ion_mod=ion_mod, mods=mods)
        ions_mz[idx] = ion_mz
        idx += 1
    if append_pep:
        aa = peptide[n_aa - 1]
        neutral_mass += AA_MASS[aa]
        if mods is not None and n_aa - 1 in mods_dict:
            neutral_mass += PTM[mods_dict[n_aa-1]]
        #if ion_type == 'y':
        #    ion_mz = cal_ion_mz(pre_neutral_mass - H2O -
        #                        neutral_mass, charge, ion_type, ion_mod=ion_mod, mods=mods)
        
        ion_mz = cal_ion_mz(neutral_mass, charge, ion_type, ion_mod=ion_mod, mods=mods)
        ions_mz[idx] = ion_mz
        idx += 1
    if append_proton:
       ions_mz[idx] = PROTON
       idx += 1

    return ions_mz


def cal_sliding_pep_mass(peptide, mods: Optional[Tuple[Tuple]] = None, 
                         context_len: int = 32, 
                         append_proton: bool = False,
                         append_pep: bool = False):
    #TODO: finish this function
    curr_mass = 0
    rev_mass = 0
    sliding_mass = [0] * (context_len + 2)
    rev_sliding_mass = [0] * (context_len + 2)
    n_aa = len(peptide)
    for i in range(n_aa):
        aa = peptide[i]
        rev_aa = peptide[n_aa - 1 - i]
        sliding_mass[i + 1] = (curr_mass + AA_MASS[aa])
        rev_sliding_mass[i + 1] = (rev_mass + AA_MASS[rev_aa])
        curr_mass += AA_MASS[aa]
        rev_mass += AA_MASS[rev_aa]
    return sliding_mass, rev_sliding_mass


def mass_comparator(pep_a, pep_b):
    mass_a = cal_pep_mass(pep_a)
    mass_b = cal_pep_mass(pep_b)
    if mass_a > mass_b:
        return 1
    elif mass_a == mass_b:
        return 0
    else:
        return -1
