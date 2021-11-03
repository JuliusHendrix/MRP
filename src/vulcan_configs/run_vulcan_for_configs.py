import glob
import os
import sys
from tqdm import tqdm
import shutil
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import random
import time

# to suppress output
# from https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def make_vulcan_copies(num_workers, VULCAN_dir):
    """
    make num_workers copies of VULCAN dir and return a list of their directories.
    """

    print(f'making {num_workers} copies of VULCAN...')

    copies_base_dir = os.path.expanduser('~/git/vulcans/')

    # remake the folder
    if os.path.isdir(copies_base_dir):
        shutil.rmtree(copies_base_dir)
    os.mkdir(copies_base_dir)

    # make list of all available dirs
    copy_dir_list = []

    # make copies
    for i in tqdm(range(num_workers)):
        copy_dir = os.path.join(copies_base_dir, f'VULCAN_{i}')
        shutil.copytree(VULCAN_dir, copy_dir)
        copy_dir_list.append(copy_dir)

    return copy_dir_list


class CopyManager:
    """
    Manage available VULCAN copies for multiprocessing.
    """
    def __init__(self, num_workers, VULCAN_dir):
        # create VULCAN copies and save their directories
        self.available_copies = make_vulcan_copies(num_workers, VULCAN_dir)

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
    (config_file, copy_manager) = params

    # get available VULCAN dir copy
    available_dir = None
    while available_dir is None:
        available_dir = copy_manager.get_available_copy()
        time.sleep(1)

    # change working directory of this process
    os.chdir(available_dir)
    sys.path.append(available_dir)
    print(f'worker on {os.getcwd()}')

    # copy config file to VULCAN directory
    shutil.copyfile(config_file, os.path.join(available_dir, 'vulcan_cfg.py'))

    # run VULCAN
    # with suppress_stdout():
    exec(open(os.path.join(available_dir, "vulcan.py")).read())

    # add VULCAN dir copy back to list
    copy_manager.add_used_copy(available_dir)

    return 0


def main(batch=False, batch_size=100, parallel=True):
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, 'configs/')
    VULCAN_dir = os.path.expanduser('~/git/VULCAN/')
    output_dir = os.path.expanduser('~/git/MRP/vulcan_configs/vulcan_output')    # TODO: change to /data
    # output_dir = os.path.expanduser('~/git/MRP/data/vulcan_output')

    # remake output directory
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

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
        if batch_size < mp.cpu_count() - 1:
            num_workers = batch_size
        else:
            num_workers = mp.cpu_count() - 1

        # setup copy manager
        BaseManager.register('CopyManager', CopyManager)
        manager = BaseManager()
        manager.start()
        mp_copy_manager = manager.CopyManager(num_workers, VULCAN_dir)

        # make mp params
        mp_params = [(cf, mp_copy_manager) for cf in config_files]

        # run mp Pool
        print(f'running VULCAN for configs with {num_workers} workers...')
        with mp.Pool(num_workers) as p:
            results = list(tqdm(p.imap(run_vulcan, mp_params),  # return results otherwise it doesn't work properly
                                total=len(mp_params)))
    else:
        # if sequential, only 1 worker
        copy_manager = CopyManager(num_workers=1, VULCAN_dir=VULCAN_dir)

        # run sequentially
        print('running VULCAN for configs sequentially...')
        for params in tqdm(config_files):
            run_vulcan((params, copy_manager))


if __name__ == "__main__":
    main(batch=True,
         batch_size=4,
         parallel=True)
