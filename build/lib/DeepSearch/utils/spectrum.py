# class for different types of ions
import re
from typing import Tuple, NamedTuple
import numpy as np
import torch


def binning_spectrum_tensor(spectrum: torch.Tensor, start_mz: float, end_mz: float, mz_step: float) -> torch.Tensor:
    """binning_spectrum_tensor _summary_

    Args:
        spectrum (torch.Tensor): _description_
        start_mz (float): _description_
        end_mz (float): _description_
        mz_step (float): _description_

    Returns:
        torch.Tensor: _description_
    """
    n_bin = int((end_mz - start_mz) / mz_step)
    intensities = spectrum[..., 1].unsqueeze(1)
    mz = spectrum[..., 0].unsqueeze(1)
    segments = ((mz - start_mz)/mz_step).long()
    assert segments.shape == intensities.shape

    binned = torch.zeros((n_bin, 1), dtype=spectrum.dtype)
    binned.scatter_reduce_(0, segments, intensities, "amax")

    return binned


# binning from padded 2d spectrum tensor, normalized intensity
def binning_spectrum_tensor_batch(spectrum: torch.Tensor, start_mz: float, end_mz: float, mz_step: float) -> torch.Tensor:
    """binning_spectrum_tensor _summary_

    Args:
        spectrum (torch.Tensor): _description_
        start_mz (float): _description_
        end_mz (float): _description_
        mz_step (float): _description_

    Returns:
        torch.Tensor: _description_
    """
    n_bin = int((end_mz - start_mz) / mz_step)
    intensities = spectrum[...,1,...].unsqueeze(2)
    mz = spectrum[...,0,...].unsqueeze(2)
    
    batch_size = spectrum.shape[0]
    segments = ((mz - start_mz)/mz_step).long()
    assert segments.shape == intensities.shape

    binned = torch.zeros((batch_size, n_bin, 1), dtype=spectrum.dtype, device=spectrum.device)
    binned.scatter_reduce_(1, segments, intensities, "amax")
    return binned


def binning_spectrum(mz_array: np.ndarray, intensity_array: np.ndarray, start_mz, end_mz, mz_step):
    n_bin = int((end_mz - start_mz) / mz_step)
    max_intensity = np.max(intensity_array)
    bin_intensities = [0.] * n_bin
    bin_mz_remainder = [0.] * n_bin

    segments = (mz_array - start_mz)/mz_step
    segments = segments.astype(int)
    mz_remainder = (mz_array - start_mz) % mz_step

    u_seg = set(segments.tolist())
    # print(segments)
    # print(end_mz)
    for segment in u_seg:
        if segment >= n_bin:
            continue
        index = np.where(segments == segment)[0]
        idx = np.argmax(intensity_array[index])
        bin_intensities[segment] = intensity_array[index][idx]
        bin_mz_remainder[segment] = mz_remainder[index][idx]

    bin_intensities = np.array(bin_intensities) / max_intensity
    bin_mz_remainder = np.array(bin_mz_remainder)
    return bin_intensities, bin_mz_remainder

#binning_spectrum(np.array([150., 200.0]), np.array([3., 4.]), 100, 250, 0.5)

def reverse_binning(binned_spectrum: np.ndarray, start_mz, end_mz, mz_step):
    n_bin = int((end_mz - start_mz) / mz_step)
    assert n_bin == binned_spectrum.shape[1]
    binned_intensity = binned_spectrum[0]
    mz_remainder = binned_spectrum[1]

    mzs = []
    intensities = []

    for i in range(n_bin):
        if binned_intensity[i] == 0.:
            continue
        mz = mz_step * i + mz_remainder[i] + start_mz
        mzs.append(mz)
        intensities.append(binned_intensity[i])
    
    return mzs, intensities



class Ion(NamedTuple):
    ion_class:  str
    index:      int 
    charge:     int
    n_neutron:  int 


# base spectrum


class Spectrum:
    def __init__(self):
        self.pre_mass = 0.
        self.pre_mz = 0.
        self.charge = 0
        self.n_peaks = 0
        self.HCD_energy = 0
        self.raw_file = ""
        self.scan = 0
        self.mz_array = []
        self.intensity_array = []
        # self.norm_intensity_array = []


# MSP Spectrum
class MSPSpectrum(Spectrum):
    """ MSPSpectrum
        params:
            pre_mass:   precursor mass
            pre_mz:     precursor m/z
            charge:     number of charges
            n_mods:     number of modifications
            n_peaks:    number of peaks
            HCD_energy: HCD collision energy
            raw_file:   corresponding raw spectrum file
            scan:       scan id
            ori_seq:    unmodified peptide sequence
            mod_seq:    modified peptide sequence
            name:       name of MSP spectrum
    """

    def __init__(self, name: str):
        super().__init__()
        self.n_mods = 0
        self.ori_seq = ""
        self.mod_seq = ""
        self.name = name
        self.annotations = []
        self.type = ""
        self.mods = set()
        self.delta_mz = []

        self.ori_seq = self.name.split('/')[0]

    def _parse_ion_type(self, ion: str):
        """_parse_ion_type _summary_

        Args:
            ion (str): _description_

        Returns:
            _type_: _description_
        """
        ion_infos = list(filter(None, re.split('\-|\+|\^', ion)))

        # class of ion (a, b, y)
        ion_class = ion_infos[0][0]
        # index
        ion_index = int(ion_infos[0][1:])
        # charge of ion
        ion_charge = 1
        # -NH3 +H2O +/-CO
        module_list = []
        # isotopic info
        num_neutron = 0

        modules = ion_infos[1:]

        for module in modules:
            # found loss of NH3
            if module == "NH3":
                module_list.append(module)
            # found loss of H2O
            elif module == "H2O":
                module_list.append(module)
            # charge
            elif module.isdigit():
                ion_charge = int(module)
            # isotopic info
            elif module[-1] == 'i':
                if len(module) == 1:
                    num_neutron == 1
                else:
                    num_neutron = int(module[0])
            # encounter other modules, unknown
            else:
                return "U"
        # multiple neutral loss
        if len(module_list) > 1:
            return "U"
        elif len(module_list) == 1:
            ion_class = ion_class + "-" + module_list[0]
        return Ion(ion_class, ion_index, ion_charge, num_neutron)

    def _parse_annotation(self, annotation: str) -> Tuple:
        """_parse_annotation _summary_

        Args:
            annotation (str): _description_

        Returns:
            tuple[Ion, float]: _description_
        """
        # unexpected internal fragmentations
        if annotation.startswith("Int"):
            _, _, delta = annotation.strip().split('/')
            return "I", float(delta.split("ppm")[0])
        # unfragmented peptide ion
        elif annotation.startswith("p"):
            _, delta = annotation.strip().split('/')
            return "P", float(delta.split("ppm")[0])

        # a, b, y ion, consider charges, isotopic and neutral losses
        elif annotation.startswith("y") or annotation.startswith("b") or annotation.startswith("a"):
            ion, delta = annotation.strip().split('/')
            ion = self._parse_ion_type(ion)
            return ion, float(delta.split("ppm")[0])
        # unknown, low priority
        else:
            return "U", 10.

    def _parse_mod_str(self, mods_strs: str):
        """parse_mod_str parse modification str and set self.mod_seq
            only consider M->m oxidation and C->c CAM

        Args:
            mods_strs (str): _description_
        """
        mods = list(filter(None, re.split('\(|\)', mods_strs)))
        self.mod_seq = [c for c in self.ori_seq]

        self.n_mods = int(mods[0])
        # no modiciation
        if self.n_mods == 0:
            return

        mods = mods[1:]
        assert len(mods) == self.n_mods

        for mod_str in mods:
            idx, aa, mod = mod_str.split(",")
            idx = int(idx)
            assert self.mod_seq[idx] == aa
            self.mod_seq[idx] = aa.lower()
            self.mods.add(mod)
        

    def parse_annotations(self, annotation_list) -> None:
        """parse_annotations _summary_

        Args:
            annotation_list (_type_): _description_
        """

        for annotations in annotation_list:
            if annotations is None:
                self.annotations.append(None)
                self.delta_mz.append(None)
            else:
                annotations = annotations.strip().split(',')

                min_ppm = 100.
                best_ion = None

                # determine most possible annotations if multiple available
                # regular ion type will have higher priority
                for annotation in annotations:
                    ion, ppm = self._parse_annotation(annotation)
                    if abs(ppm) < abs(min_ppm):
                        best_ion = ion
                        min_ppm = ppm

                self.annotations.append(best_ion)
                self.delta_mz.append(ppm)
        # logging.debug({self.annotations})
        return

    def parse_comment(self, comment_str: str):
        """parse_comment _summary_

        Args:
            comment_str (str): _description_
        """
        def parse_info(var: str):
            vars = var.split("=")
            if len(vars) != 2:
                return None, None
            return vars[0], vars[1]

        comments = comment_str.strip().split()[1:]

        for comment in comments:
            key, val = parse_info(comment)
            if key == "Pep":
                self.type = val
            elif key == "Mods":
                self._parse_mod_str(val)
            elif key == "Charge":
                self.charge = int(val)
            elif key == "Parent":
                self.pre_mz = float(val)
            elif key == "HCD":
                self.HCD_energy = float(val.strip("%"))
            elif key == "Scan":
                self.scan = int(val)
            elif key == "Origfile":
                self.raw_file = val
            elif key == "Fullname":
                assert self.ori_seq == val[2:-2]

    def parse_peaks(self, mzs, intensities, annotations):
        """parse_peaks parse m/z, intensities, normalized intensities and annotations

        Args:
            mzs (_type_): _description_
            intensities (_type_): _description_
            annotations (_type_): _description_
        """
        self.mz_array = list(mzs)
        self.intensity_array = list(intensities)

        max_intensity = max(self.intensity_array)
        #for intensity in self.intensity_array:
        #    self.norm_intensity_array.append(intensity/max_intensity)

        self.parse_annotations(annotations)
        
