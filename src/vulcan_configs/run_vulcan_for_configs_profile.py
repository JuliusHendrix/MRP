import glob
import importlib
import os
import sys
import numpy as np
from tqdm import tqdm
import shutil
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import random
import time
from contextlib import redirect_stdout, redirect_stderr
import cProfile
import psutil

# TODO: don't know if this is nescessary
# Limiting the number of threads
os.environ["OMP_NUM_THREADS"] = "1"


class CopyManager:
    """
    Manage available VULCAN copies for multiprocessing.
    """

    def __init__(self, num_workers, VULCAN_dir):
        self.VULCAN_dir = VULCAN_dir
        self.copies_base_dir = os.path.expanduser('~/git/vulcans/')

        # create VULCAN copies and save their directories
        self.available_copies = self.make_initial_copies(num_workers)

    def make_initial_copies(self, num_workers):
        """
        Make num_workers copies of the VULCAN directory.
        """
        print(f'making {num_workers} copies of VULCAN...')

        # remake the folder
        if os.path.isdir(self.copies_base_dir):
            shutil.rmtree(self.copies_base_dir)
        os.mkdir(self.copies_base_dir)

        # make list of all available dirs
        copy_dir_list = []

        # make copies
        for i in tqdm(range(num_workers)):
            copy_dir = os.path.join(self.copies_base_dir, f'VULCAN_{i}')
            shutil.copytree(self.VULCAN_dir, copy_dir)
            copy_dir_list.append(copy_dir)

        return copy_dir_list

    def get_available_copy(self):
        """
        Get a copy directory from the list of available directories
        """

        if len(self.available_copies) > 0:
            available_copy = self.available_copies[0]
            self.available_copies.remove(available_copy)
        else:
            available_copy = None

        return available_copy

    def add_used_copy(self, used_copy):
        """
        Add a directory to the list of available directories
        """

        if used_copy not in self.available_copies:
            self.available_copies.append(used_copy)
        else:
            raise ValueError(f'{used_copy} already in available_copies!')


def run_vulcan(params):
    (config_file, copy_manager, std_output_dir) = params

    # get available VULCAN dir copy
    available_dir = None
    while available_dir is None:
        available_dir = copy_manager.get_available_copy()
        if available_dir is None:
            print('copy dir not available')
            time.sleep(1)

    # change working directory of this process
    os.chdir(available_dir)
    sys.path.append(available_dir)

    # copy config file to VULCAN directory
    shutil.copyfile(config_file, os.path.join(available_dir, 'vulcan_cfg.py'))

    # make std_output redirect file
    cf_name = os.path.basename(config_file)
    std_output_file = os.path.join(std_output_dir, f'{cf_name[:-3]}.txt')

    # print info
    print(
        f'\n{mp.current_process()}'
        f'\non cpu {psutil.Process().cpu_num()}'
        f'\nin {os.getcwd()}'
        f'\nwith {os.path.basename(config_file)}\n'
    )

    # save output to file
    with open(std_output_file, 'a+') as f:
        with redirect_stdout(f):
            with redirect_stderr(f):
                start = time.time()  # start timer

                # run VULCAN
                if 'vulcan' in sys.modules.keys():  # checks if vulcan has been imported already,
                    importlib.reload(vulcan)  # and if so, it reimports
                else:
                    import vulcan

                # exec(open(os.path.join(available_dir, "vulcan.py")).read())    # run VULCAN

                duration = (time.time() - start) / 60.
                print(f'\nVULCAN run took {duration} minutes')  # save time

    # add VULCAN dir copy back to list
    copy_manager.add_used_copy(available_dir)

    return duration


# make profile file for all workers
def profile_worker(mp_params):
    # extract parameters
    (i, cf, mp_copy_manager, std_output_dir, profile_dir) = mp_params
    params = (cf, mp_copy_manager, std_output_dir)

    # profiling
    profile_file = os.path.join(profile_dir, f'profile-{i}.out')

    print(f'profiling in {profile_file}...')
    cProfile.runctx('run_vulcan(params)', globals(), locals(), profile_file)


def main(batch=False, batch_size=100, parallel=True):
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, '../../data/configs/')
    VULCAN_dir = os.path.expanduser('~/git/VULCAN/')
    output_dir = os.path.expanduser('/data/vulcan_output')
    std_output_dir = os.path.join(output_dir, 'std_output/')
    profile_dir = os.path.join(script_dir, f'profile_{batch_size}_workers/')

    # remake output directory
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # remake std_output directory
    if os.path.isdir(std_output_dir):
        shutil.rmtree(std_output_dir)
    os.mkdir(std_output_dir)

    # remake profiling directory
    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir)
    os.mkdir(profile_dir)

    # load config files
    config_files = glob.glob(os.path.join(configs_dir, 'vulcan_cfg*.py'))
    print(f'found {len(config_files)} config files')

    # create random batch of config files
    if batch:
        print(f'using random batch of {batch_size} configs')
        batch_files = random.sample(config_files, batch_size)
        config_files = batch_files

    if parallel:
        # number of processes
        num_workers = batch_size

        # setup copy manager
        BaseManager.register('CopyManager', CopyManager)
        manager = BaseManager()
        manager.start()
        mp_copy_manager = manager.CopyManager(num_workers, VULCAN_dir)

        # make mp params
        mp_params = [(i, cf, mp_copy_manager, std_output_dir, profile_dir) for i, cf in enumerate(config_files)]

        # run mp Pool
        print(f'running VULCAN for configs with {num_workers} workers...')
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(pool.imap(profile_worker, mp_params))

    else:
        # if sequential, only 1 copy
        copy_manager = CopyManager(num_workers=1, VULCAN_dir=VULCAN_dir)

        # run sequentially
        print('running VULCAN for configs sequentially...')
        for params in tqdm(config_files):
            run_vulcan((params, copy_manager, std_output_dir))


if __name__ == "__main__":
    batch_sizes = [8]

    for batch_size in batch_sizes:
        main(batch=True,
             batch_size=batch_size,
             parallel=True)
