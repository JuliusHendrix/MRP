import glob
import os
import sys
from tqdm import tqdm
import shutil
from contextlib import contextmanager
import multiprocessing as mp

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


def run_vulcan(params):
    (config_file, VULCAN_dir) = params
    # copy config file to VULCAN directory
    shutil.copyfile(config_file, os.path.join(VULCAN_dir, 'vulcan_cfg.py'))

    # run VULCAN
    with suppress_stdout():
        exec(open(os.path.join(VULCAN_dir, "vulcan.py")).read())

    return 0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, 'configs')
    VULCAN_dir = os.path.expanduser('~/git/VULCAN/')

    # Change the current working directory
    os.chdir(VULCAN_dir)
    sys.path.append(VULCAN_dir)

    config_files = glob.glob(os.path.join(configs_dir, '*.py'))

    mp_params = [(cf, VULCAN_dir) for cf in config_files]
    num_workers = mp.cpu_count() - 1

    print('running VULCAN for configs...')
    with mp.Pool(num_workers) as p:
        results = list(tqdm(p.imap(run_vulcan, mp_params),  # return results otherwise it doesn't work properly
                            total=len(mp_params)))


if __name__ == "__main__":
    main()
