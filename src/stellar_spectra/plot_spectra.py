import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, 'images')

# create image dir
if os.path.isdir(image_dir):
    shutil.rmtree(image_dir)
os.mkdir(image_dir)

spectrum_files = glob.glob(os.path.join(script_dir, 'output/*_K.txt'))

for spectrum_file in spectrum_files:
    spectrum = np.genfromtxt(spectrum_file, dtype=float, skip_header=1, names = ['lambda', 'flux'])

    filename = os.path.basename(spectrum_file)
    # print(filename)
    # print(spectrum['flux'].shape)
    # print(spectrum['lambda'].shape)

    plt.figure(figsize=(6,4))
    plt.title(filename)
    plt.plot(spectrum['lambda'], spectrum['flux'], linewidth=1, color='darkblue')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('flux (erg / (nm cm2 s))')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(image_dir, f'{filename}.png'), dpi=600)
    plt.show()
    break
