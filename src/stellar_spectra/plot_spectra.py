import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, 'images')

colors = [
'#065e60', '#0a9396', '#99d5c9', '#ee9b00', '#ffb833', '#99d6ff', '#0093f5', '#00568f'
]

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# create image dir
if os.path.isdir(image_dir):
    shutil.rmtree(image_dir)
os.mkdir(image_dir)

spectrum_files = glob.glob(os.path.join(script_dir, 'output/*_K.txt'))

for spectrum_file in spectrum_files:
    spectrum = np.genfromtxt(spectrum_file, dtype=float, skip_header=1, names = ['lambda', 'flux'])

    filename = os.path.basename(spectrum_file)

    T = float(filename.split('_')[0])
    # print(filename)
    # print(spectrum['flux'].shape)
    # print(spectrum['lambda'].shape)

    spectrum['flux'] = np.where(spectrum['flux'] < 1e-10, 1e-10, spectrum['flux'])

    plt.figure(figsize=(5,4))
    plt.title(
        'Stellar Spectrum\n'
        rf'T$_{{eff}}$ = {T:.0f} K')
    plt.plot(spectrum['lambda'], spectrum['flux'], linewidth=1, color=colors[1])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(r'Flux [erg nm$^{-1}$ cm$^{-2}$ s$^{-1}]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(
        os.path.join(image_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    # break
