import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import sys
import os
import shutil
from astropy.io import fits

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output/')
# TODO: change to MRP/data folder?
pRT_input_data_path = os.path.join(script_dir, 'input_data_std/input_data')

# export petitRADTRANS input path
os.environ["pRT_input_data_path"] = str(pRT_input_data_path)

# create output dir
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

sys.path.append(os.path.join(script_dir, '../petitRADTRANS-master/'))
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import h5py

# Setting the temperature range 
TEMP_grid = np.linspace(3000,6000,61)

# Constants
au = 1.4959787e13
r_sun = 6.957e10

# Functions to open and save hdf5 files
def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def specs_list():
    '''
		Reading in MUSCLES star characteristics
	'''

    file = '../stellar_spectra/MUSCLES_stars.dat'

    dat = open(file, 'r')

    stars = {}
    for line in dat:
        if line.startswith('#'): continue
        d = re.sub("[^\w]", " ", line).split()
        d[1] = float(d[1])
        if len(d) == 5:
            d[3] = d[3] + '.' + d[4]
        if len(d) == 6:
            d[2] = d[2] + '.' + d[3]
            d[3] = d[4] + '.' + d[5]
        d[2] = float(d[2])
        d[3] = float(d[3])
        stars[d[0].lower()] = d[1:4]

    return stars


def read_MUSCLES():
    '''
		Reading in MUSCLES data
	'''

    MUS = specs_list()

    NAMES = list(MUS.keys())
    for i in range(len(NAMES)):

        file_full = 'stellar_specs/hlsp_muscles_multi_multi_{0}_broadband_v22_adapt-const-res-sed.fits'.format(NAMES[i])

        if NAMES[i] == 'gj551':
            file_HST = 'stellar_specs/hlsp_muscles_hst_stis_{0}_g750l_v22_component-spec.fits'.format(NAMES[i])
        elif NAMES[i] == 'vepseri':
            file_HST = 'stellar_specs/hlsp_muscles_hst_stis_{0}_g430m_v22_component-spec.fits'.format(NAMES[i])
        else:
            file_HST = 'stellar_specs/hlsp_muscles_hst_stis_{0}_g430l_v22_component-spec.fits'.format(NAMES[i])

        spec_full = fits.getdata(os.path.join(script_dir, file_full), 1)
        flux_full = spec_full['FLUX']
        wavs_full = spec_full['WAVELENGTH']

        spec_HST = fits.getdata(os.path.join(script_dir, file_HST), 1)
        flux_HST = spec_HST['FLUX']
        wavs_HST = spec_HST['WAVELENGTH']

        flux_full = flux_full[np.where(wavs_full < max(wavs_HST))[0]]
        wavs_full = wavs_full[np.where(wavs_full < max(wavs_HST))[0]]

        MUS[NAMES[i]].append(flux_full)
        MUS[NAMES[i]].append(wavs_full)

    return MUS


def create_specs(TEMP_grid, save_to_txt=False, multiprocessing_lock=None):
    '''
		Function to create the stellar spectra with
	'''
    MUS = read_MUSCLES()

    Temps_MUS = np.array(list(MUS.values()), dtype=object)[:, 0]
    names_MUS = list(MUS.keys())

    spec_diff = np.zeros(len(TEMP_grid))
    T_diff = np.zeros_like(spec_diff)

    # Looping over all temperature points
    for i in range(len(TEMP_grid)):
        T = TEMP_grid[i]

        Tdiff_indx = np.argmin(abs(Temps_MUS - T))

        stellar_spec = nc.get_PHOENIX_spec(T)

        # If the difference in MUSCLES star temperature and evaluated temperature is too big,
        # simply use the PHOENIX spectra solely
        if abs(Temps_MUS[Tdiff_indx] - T) > 1000:
            # print('Using PHOENIX')
            wlen = stellar_spec[:, 0]  # *1e8
            flux_star = stellar_spec[:, 1]
            flux_star = flux_star * (3e10) / (wlen ** 2)
            wlen = stellar_spec[:, 0] * 1e8
            flux_star = flux_star * 1e-8

            flux_total = np.copy(flux_star)
            wavs_total = np.copy(wlen)

            T_diff[i] = 1
            spec_diff[i] = 1

            # continue

        # Otherwise stitch the MUSCLES UV part with the PHOENIX spectrum
        else:
            # print('Stitching MUSCLES UV')
            wlen = stellar_spec[:, 0]  # *1e8
            flux_star = stellar_spec[:, 1]
            flux_star = flux_star * (3e10) / (wlen ** 2)
            wlen = stellar_spec[:, 0] * 1e8
            flux_star = flux_star * 1e-8

            name_MUS = names_MUS[Tdiff_indx]
            flux_MUS = MUS[name_MUS][3] * ((63241 * au * MUS[name_MUS][1] * 3.262 / (r_sun * MUS[name_MUS][2])) ** 2)
            wavs_MUS = MUS[name_MUS][4]

            flux_total = np.concatenate((flux_MUS, flux_star[np.where(wlen > max(wavs_MUS))[0]]))
            wavs_total = np.concatenate((wavs_MUS, wlen[np.where(wlen > max(wavs_MUS))[0]]))

            T_diff[i] = Temps_MUS[Tdiff_indx] - T
            spec_diff[i] = get_diff(flux_MUS, flux_star[np.where(wlen > max(wavs_MUS))[0]])

            # If the PHOENIX spectrum is more than 2 times bigger than the MUSCLES spectrum
            # at the same wavelength point (where they are stitched), also use the PHOENIX
            # spectrum solely
            if spec_diff[i] > 2:
                # print('Spectral difference too big! - switching back to PHOENIX')
                flux_total = np.copy(flux_star)
                wavs_total = np.copy(wlen)
                T_diff[i] = 1
                spec_diff[i] = 1

            # Saving the spectrum to txt file such that it can be used straight away in
        # VULCAN
        if save_to_txt:

            wavs = wavs_total * 0.1
            flux = flux_total * 10

            new_str = 'Wavelength (nm),' + '\t' + 'Flux (erg / (nm cm2 s))' + '\n'

            for j in range(len(wavs)):

                if wavs[j] > 10000.:
                    break
                new_str += str(wavs[j]) + '\t' + str(flux[j]) + '\n'

            txt_file = os.path.join(output_dir, f'{T}_K.txt')

            if not os.path.isfile(txt_file):
                with open(txt_file, 'w') as f:
                    f.write(new_str)

    return txt_file


def get_diff(spec_A, spec_B):
    '''
		Function to get ratio between Phoenix and Muscles, can
			be used to set threshold
	'''

    # Below function not used but can be if needed
    N = 1
    A = np.sum(spec_A[-N:]) / len(spec_A[-N:])
    B = np.sum(spec_B[:N]) / len(spec_B[:N])

    # This one is used
    A = spec_A[-1]
    B = spec_B[0]

    return B / A


def main():
    create_specs(save_to_txt=True)


if __name__ == "__main__":
    sys.exit(main())
